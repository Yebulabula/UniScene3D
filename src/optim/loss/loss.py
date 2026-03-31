"""Base loss registry and composition helpers."""

import torch.nn as nn
import torch.nn.functional as F
from fvcore.common.registry import Registry

LOSS_REGISTRY = Registry("loss")

def answer_loss(data_dict):
    return F.binary_cross_entropy_with_logits(
            data_dict["answer_scores"], data_dict["answer_label"].float(), reduction='sum'
        ) / data_dict["answer_scores"].shape[0]

class Loss(nn.Module):
    def __init__(self, cfg, accelerator):
        super().__init__()
        self.all_keys = list(set(cfg.model.vis_loss_list + cfg.model.loss_list))

        self.loss_fn = {}
        for k in self.all_keys:
            if k in globals().keys():
                self.loss_fn[k] = globals()[k]
                print(f"Using {k} from loss.globals()")
            else:
                self.loss_fn[k] = LOSS_REGISTRY.get(k)(cfg, accelerator)
                setattr(self, k, self.loss_fn[k]) # register the loss module, otherwise its parameters will not be the same device as the model
                print(f"Using {k} from Registry {LOSS_REGISTRY._name}")

    def forward(self, data_dict):
        all_losses = {}

        for k, fn in self.loss_fn.items():
            # Compute current loss
            cur_loss = fn(data_dict)

            if isinstance(cur_loss, dict):
                all_losses.update(cur_loss)
            else:
                all_losses[k] = cur_loss

        total_loss = sum(all_losses.values())
        all_losses["total_loss"] = total_loss

        return total_loss, all_losses
