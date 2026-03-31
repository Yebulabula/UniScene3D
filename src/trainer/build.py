"""Trainer registry and trainer setup logic."""

import glob
from datetime import timedelta
from pathlib import Path
from omegaconf import OmegaConf
import numpy as np

from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed, InitProcessGroupKwargs
from fvcore.common.registry import Registry
import torch
import wandb

import common.misc as misc
from data.build import build_dataloader
from evaluator.common.build import build_eval
from model.build import build_model
from optim.build import build_optim
from safetensors.torch import load_file

TRAINER_REGISTRY = Registry("Trainer")


class Tracker():
    def __init__(self, cfg):
        self.reset(cfg)

    def step(self):
        self.epoch += 1

    def reset(self, cfg):
        self.exp_name = f"{cfg.exp_dir.parent.name.replace(f'{cfg.name}', '').lstrip('_')}/{cfg.exp_dir.name}"
        self.epoch = 0
        self.best_result = -np.inf

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if not k.startswith('__')}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

@TRAINER_REGISTRY.register()
class BaseTrainer():
    def __init__(self, cfg):
        set_seed(cfg.rng_seed)
        self.debug = cfg.debug.get("flag", False)
        self.hard_debug = cfg.debug.get("hard_debug", False)
        self.epochs_per_eval = cfg.solver.get("epochs_per_eval", None)
        self.epochs_per_save = cfg.solver.get("epochs_per_save", None)
        self.global_step = 0

        self.exp_tracker = Tracker(cfg)
        wandb_args = {"entity": cfg.logger.entity, "id": cfg.logger.run_id, "resume": cfg.resume}
        if not cfg.logger.get('autoname'):
            wandb_args["name"] = self.exp_tracker.exp_name
        self.logger = get_logger(__name__)
        self.mode = cfg.mode

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
        kwargs = ([ddp_kwargs] if cfg.num_gpu > 1 else []) + [init_kwargs]

        gradient_accumulation_steps = cfg.solver.get("gradient_accumulation_steps", 1)

        self.accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_with=cfg.logger.name,
            kwargs_handlers=kwargs,
        )

        if not self.hard_debug:
            self.accelerator.init_trackers(
                project_name=cfg.name if not self.debug else "Debug",
                config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) if not cfg.resume else None,
                init_kwargs={"wandb": wandb_args},
            )

        print(OmegaConf.to_yaml(cfg))

        if self.mode == "pretrain":
            keys = ["pretrain"]
        else:
            keys = ["train", "val", "test"]

        # Build data first so model and optimizer can use dataloader sizes.
        self.data_loaders = {key: build_dataloader(cfg, split=key) for key in keys}
        self.model = build_model(cfg)
        self.epochs = cfg.solver.epochs

        if self.mode == "test":
            total_steps = 1
        else:
            total_steps = (len(self.data_loaders[self.mode]) * self.epochs) // gradient_accumulation_steps
        self.loss, self.optimizer, self.scheduler = build_optim(
            cfg,
            self.model.get_opt_params(),
            total_steps=total_steps,
            accelerator=self.accelerator,
        )

        if self.mode == "pretrain":
            self.evaluator = None
        else:
            if misc.rgetattr(cfg, "eval.pass_kwargs", False):
                kwargs = {"dataloaders": self.data_loaders}
            else:
                kwargs = {}
            self.evaluator = build_eval(cfg, self.accelerator, **kwargs)

        self.total_steps = 1 if self.mode == "test" else len(self.data_loaders[self.mode]) * self.epochs
        self.grad_norm = cfg.solver.get("grad_norm")

        ema = [0.996, 1.0]
        ipe_scale = 1.0
        self.momentum_scheduler = (
            ema[0] + i * (ema[1] - ema[0]) / (self.total_steps * self.epochs * ipe_scale)
            for i in range(int(self.total_steps * self.epochs * ipe_scale) + 1)
        )

        if cfg.get('pretrain_ckpt_path'):
            self.pretrain_ckpt_path = Path(cfg.pretrain_ckpt_path)
            self.load_pretrain()
            # Keep the point-map embedding in sync with the RGB patch embedding if requested.
            if hasattr(self.model, "sync_geo_embedding_from_patch_after_load"):
                self.model.sync_geo_embedding_from_patch_after_load()
            if hasattr(self.model, "pm_encoder"):
                self.model.pm_encoder.load_state_dict(self.model.pm_encoder.state_dict())

        # Let Accelerate wrap the model, optimizer, and dataloaders for the chosen backend.
        self.model, self.loss, self.optimizer, self.scheduler = self.accelerator.prepare(
            self.model, self.loss, self.optimizer, self.scheduler
        )
        for name, loader in self.data_loaders.items():
            if isinstance(loader, list):
                loader = self.accelerator.prepare(*loader)
            else:
                loader = self.accelerator.prepare(loader)
            self.data_loaders[name] = loader
        self.accelerator.register_for_checkpointing(self.exp_tracker)

        self.ckpt_path = Path(cfg.ckpt_path) if cfg.get("ckpt_path") else Path(cfg.exp_dir) / "ckpt" / "best.pth"
        if cfg.resume:
            self.resume()

    def forward(self, data_dict):
        return self.model(data_dict)

    def update_ema(self):
        with torch.no_grad():
            m = next(self.momentum_scheduler)
            model_context = self.model.module.context_model if hasattr(self.model, 'module') else self.model.context_model
            model_target = self.model.module.target_model if hasattr(self.model, 'module') else self.model.target_model

            for param_q, param_k in zip(model_context.parameters(), model_target.parameters()):
                param_k.data.mul_(m).add_((1.0 - m) * param_q.detach().data)

    def backward(self, loss):
        self.accelerator.backward(loss)

        total_norm = torch.norm(torch.stack([
            torch.norm(p.grad.detach()) for p in self.model.parameters() if p.grad is not None
        ]))
        print(f"grad_norm={total_norm.item():.2f}")

        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        if self.accelerator.sync_gradients:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.scheduler.step()

    def log(self, results, mode="train"):
        if not self.hard_debug:
            log_dict = {}
            for key, val in results.items():
                if isinstance(val, torch.Tensor):
                    val = val.item()
                log_dict[f"{mode}/{key}"] = val
            if mode == "train":
                lrs = self.scheduler.get_lr()
                for i, lr in enumerate(lrs):
                    log_dict[f"{mode}/lr/group_{i}"] = lr
            self.accelerator.log(log_dict, step=self.global_step)

    def save(self, name):
        misc.make_dir(self.ckpt_path.parent)
        self.save_func(str(self.ckpt_path.parent / name))

    def resume(self):
        if self.ckpt_path.exists():
            print(f"Resuming from {str(self.ckpt_path)}")
            self.accelerator.load_state(str(self.ckpt_path))
            print(f"Successfully resumed from {self.ckpt_path}")
        else:
            self.logger.info("training from scratch")

    def load_pretrain(self):
        print(f"📂 Loading pretrained weights from: {str(self.pretrain_ckpt_path)}")
        model_weight_path_pattern = str(self.pretrain_ckpt_path / "model*.safetensors")
        model_weight_paths = glob.glob(model_weight_path_pattern)

        if len(model_weight_paths) == 0:
            raise FileNotFoundError(f"❌ Cannot find any .safetensors file in {str(self.pretrain_ckpt_path)}")

        weights = {}
        for model_weight_path in model_weight_paths:
            # Merge shard files into one state dict before loading.
            weights.update(load_file(model_weight_path, device="cpu"))

        self.model.load_state_dict(weights, strict=False)

        vision_model_keys = {
            key for key in self.model.state_dict().keys()
            if key.startswith("pm_encoder.vision_model.")
        }
        loaded_vision_model_keys = vision_model_keys.intersection(weights.keys())

        if vision_model_keys and loaded_vision_model_keys == vision_model_keys:
            print(f"✅ Entire UniScene3D vision encoder is loaded.")

    def save_func(self, path):
        self.accelerator.save_state(path)

def build_trainer(cfg):
    return TRAINER_REGISTRY.get(cfg.trainer)(cfg)
