"""Optimizer factory helpers."""

import torch.optim as optim

from common.misc import cfg2dict


def get_optimizer(cfg, params):
  optimizer_cls = getattr(optim, cfg.solver.optim.name, None)
  if optimizer_cls is None:
    raise NotImplementedError(f"Unknown optimizer: {cfg.solver.optim.name}")
  return optimizer_cls(params, **cfg2dict(cfg.solver.optim.args))
