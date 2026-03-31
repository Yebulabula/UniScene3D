"""Model registry and model construction helpers."""

import torch.nn as nn
from fvcore.common.registry import Registry


MODEL_REGISTRY = Registry("model")


class BaseModel(nn.Module):
    """Base class for models built from config."""

    def __init__(self, cfg):
        """Initialize the model base class."""
        super().__init__()

    def get_opt_params(self):
        """Return parameter groups for the optimizer."""
        raise NotImplementedError("Function to obtain all default parameters for optimization")


def build_model(cfg):
    """Build a registered model from the config name."""
    model = MODEL_REGISTRY.get(cfg.model.name)(cfg)
    return model
