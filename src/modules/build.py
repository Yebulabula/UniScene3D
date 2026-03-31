"""Module and head registries."""

from fvcore.common.registry import Registry

from common.misc import cfg2dict


HEADS_REGISTRY = Registry("heads")


def build_module(module_type, cfg):
    """Build a registered module from config."""
    if module_type == "heads":
        return HEADS_REGISTRY.get(cfg.name)(**cfg2dict(cfg.args))
    else:
        raise NotImplementedError(f"module type {module_type} not implemented")
