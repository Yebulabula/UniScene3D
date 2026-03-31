"""Trainer used for UniScene3D pretraining and finetuning."""

from tqdm import tqdm
import torch
from trainer.build import TRAINER_REGISTRY, BaseTrainer

@TRAINER_REGISTRY.register()
class UniScene3DTrainer(BaseTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)

    def forward(self, data_dict, mode):
        return self.model(data_dict, mode)

    def backward(self, loss, mode=None):
        self.accelerator.backward(loss)

        if self.grad_norm is not None and self.accelerator.sync_gradients:
            self.accelerator.clip_grad_norm_(self.model.parameters(), self.grad_norm)

        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()

    def train_step(self, epoch, mode=None):
        self.model.train()
        loader = self.data_loaders[self.mode]
        is_main = self.accelerator.is_main_process

        pbar = tqdm(loader, disable=not is_main, desc=f"[Epoch {epoch + 1}/{self.epochs}]")

        for data_dict in pbar:
            with self.accelerator.accumulate(self.model):
                data_dict = self.forward(data_dict, mode=mode)
                loss, losses = self.loss(data_dict)
                self.backward(loss, mode=mode)

                self.global_step += 1
                log_dict = {'step': self.global_step, **losses}
                for key in ("drop_rgb_count", "drop_pointmap_count", "keep_both_count"):
                    if key in data_dict:
                        log_dict[key] = data_dict[key]

                if mode == 'qa':
                    metrics = self.evaluator["train"].batch_metrics(data_dict)
                    log_dict.update(metrics)
                self.log(log_dict, mode="train")

    @torch.no_grad()
    def eval_step(self, epoch, mode):
        self.model.eval()
        loader = self.data_loaders["val"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict, mode=mode)
            loss, losses = self.loss(data_dict)
            log_dict = {'epoch': epoch, **losses}
            pbar.update(1)
        self.log(log_dict, mode="val")

    @torch.no_grad()
    def test_step(self):
        self.model.eval()
        loader = self.data_loaders["test"]
        pbar = tqdm(range(len(loader)), disable=(not self.accelerator.is_main_process))
        for i, data_dict in enumerate(loader):
            data_dict = self.forward(data_dict)
            self.evaluator["val"].update(data_dict)
            pbar.update(1)
        is_best, results = self.evaluator["val"].record()
        self.log(results, mode="test")
        self.evaluator["val"].reset()
        return results

    def run(self):
        num_trainable_params = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                num_trainable_params += param.numel()
                print(name)

        print(f"Total number of trainable parameters: {num_trainable_params:,}")
        if self.mode == "pretrain":
            start_epoch = self.exp_tracker.epoch
            self.global_step = start_epoch * len(self.data_loaders[self.mode])

            for epoch in range(start_epoch, self.epochs):
                self.exp_tracker.step()
                self.train_step(epoch, mode=self.mode)

                self.accelerator.wait_for_everyone()
                if self.accelerator.is_main_process and self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                    self.save(f"ckpt_{epoch+1}.pth")

            self.save(f"ckpt_{epoch+1}.pth")
            self.accelerator.end_training()
            return

        if self.mode == "train":
            start_epoch = self.exp_tracker.epoch
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
                    if is_best:
                        self.save("best.pth")
                    if self.epochs_per_save and (epoch + 1) % self.epochs_per_save == 0:
                        self.save(f"ckpt_{epoch+1}.pth")

        self.test_step()
        if self.mode == "train":
            self.accelerator.end_training()
