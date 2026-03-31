"""Builds losses, optimizers, and schedulers."""

from optim.loss.loss import Loss
from optim.optimizer.optim import get_optimizer
from optim.scheduler import get_scheduler


def build_optim(cfg, params, total_steps, accelerator):
    loss = Loss(cfg, accelerator)
    optimizer = get_optimizer(cfg, params)
    scheduler = get_scheduler(cfg, optimizer, total_steps)
    return loss, optimizer, scheduler
