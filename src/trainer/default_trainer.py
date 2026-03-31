"""Default trainer used for downstream tasks."""

from tqdm import tqdm

import torch
from trainer.build import TRAINER_REGISTRY
from trainer.build import BaseTrainer


@TRAINER_REGISTRY.register()
class DefaultTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.best_metric = -1

    def forward(self, data_dict, mode):
        return self.model(data_dict, mode)

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.accelerator.backward(loss)
        
        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)
            
        self.optimizer.step()
        self.scheduler.step()

    def train_step(self, epoch):
        self.model.train()
        loader = self.data_loaders["train"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process), desc=f"[Epoch {epoch + 1}/{self.epochs}]")
        for i, data_dict in enumerate(loader):
            with self.accelerator.accumulate(self.model):
                data_dict['cur_step'] = epoch * len(loader) + i
                data_dict['total_steps'] = self.total_steps
                # forward
                data_dict = self.forward(data_dict, mode = 'qa')
                # calculate loss
                loss, losses = self.loss(data_dict)
                self.backward(loss)
                # record
                self.global_step += 1
                log_dict = {'step': self.global_step}
                log_dict.update(losses)
                self.log(log_dict, mode="train")
                pbar.update(1)

    def _gather_for_metrics(self, data_dict):
        out = {}
        metric_keys = [
            "answer_scores",
            "answer_label",
            "sqa_type",
            "question_type",
            "hypo3d_type",
            "sentence",
            "scan_id",
            "data_idx",
            "answers",
        ]
        for key in metric_keys:
            if key in data_dict:
                out[key] = self.accelerator.gather_for_metrics(data_dict[key])
        return out

    @torch.no_grad()
    def eval_step(self, epoch):
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))

        for _, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict, mode="qa")

            gathered = self._gather_for_metrics(data_dict)

            if self.accelerator.is_main_process:
                self.evaluator.update(gathered)

            pbar.update(1)

        self.accelerator.wait_for_everyone()

        if self.accelerator.is_main_process:
            is_best, results = self.evaluator.record()
            if is_best:
                self.best_metric = results["target_metric"]
            self.log(results, mode="val")
            self.evaluator.reset()
            return is_best

        return False

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))

        for _, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict, mode="qa")

            gathered = self._gather_for_metrics(data_dict)

            if self.accelerator.is_main_process:
                self.evaluator.update(gathered)

            pbar.update(1)

        self.accelerator.wait_for_everyone()
    
        if self.accelerator.is_main_process:
            _, results = self.evaluator.record(split="test")
            self.log(results, mode="test")
            self.evaluator.reset()
        else:
            results = None

        # broadcast results (optional). If you only need results on main, you can just return None on others.
        return results if self.accelerator.is_main_process else None

    def run(self):
        if self.mode == "train":
            model = self.model.module if hasattr(self.model, 'module') else self.model
            model.set_downstream_mode()
            start_epoch = self.exp_tracker.epoch

            num_trainable_params = 0
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    num_trainable_params += param.numel()

            print(f"Total number of trainable parameters: {num_trainable_params:,}")
            self.global_step = start_epoch * len(self.data_loaders["train"])
            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch)

                if self.epochs_per_eval and (epoch + 1) % self.epochs_per_eval == 0:
                    is_best = self.eval_step(epoch)
                    self.accelerator.print(f"[Epoch {epoch + 1}/{self.epochs}] finished eval, is_best: {is_best}")
                else:
                    is_best = False

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process:
                    self.save("latest.pth")
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")

        self.test_step()
        if self.mode == "train":
            self.accelerator.end_training()
