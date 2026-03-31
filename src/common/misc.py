"""Shared small utilities used across training and evaluation."""

import functools
import shutil
import tempfile
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
import torch
from transformers import AutoModelForCausalLM
from accelerate.logging import get_logger
from accelerate.state import PartialState
from accelerate.utils import recursively_apply
from accelerate.utils.constants import TORCH_DISTRIBUTED_OPERATION_TYPES

logger = get_logger(__name__)
DEFAULT_FGCLIP_HF_REPO_ID = "qihoo360/fg-clip-base"
DEFAULT_FGCLIP_WEIGHT_FILENAME = "model.safetensors"
DEFAULT_FGCLIP_REPO_TYPE = "model"


def make_dir(dir_path):
    if not Path(dir_path).exists():
        Path(dir_path).mkdir(parents=True, exist_ok=True)


def cfg2dict(cfg):
    return OmegaConf.to_container(cfg, resolve=True)


def _link_or_copy(src: Path, dst: Path):
    try:
        dst.symlink_to(src)
    except OSError:
        shutil.copy2(src, dst)


def build_fgclip_model_from_local_code_with_hf_weights(
    local_model_root,
    repo_id=DEFAULT_FGCLIP_HF_REPO_ID,
    weight_filename=DEFAULT_FGCLIP_WEIGHT_FILENAME,
    repo_type=DEFAULT_FGCLIP_REPO_TYPE,
):
    local_model_root = Path(local_model_root)
    weight_path = hf_hub_download(
        repo_id=repo_id,
        filename=weight_filename,
        repo_type=repo_type,
        local_files_only=False,
    )

    with tempfile.TemporaryDirectory(prefix="fgclip_hf_") as tmpdir:
        temp_root = Path(tmpdir)
        for src in local_model_root.iterdir():
            if not src.is_file() or src.name == weight_filename:
                continue
            _link_or_copy(src, temp_root / src.name)
        _link_or_copy(Path(weight_path), temp_root / weight_filename)

        model = AutoModelForCausalLM.from_pretrained(
            str(temp_root),
            trust_remote_code=True,
            local_files_only=True,
        )
    print(f"Loaded FG-CLIP weights from Hugging Face repo {repo_id}")
    return model

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))

def _gpu_gather_object(object: Any):
    # by JY Huang: re-implement the method for gathering non-tensor objects
    output_objects = [None for _ in range(PartialState().num_processes)]
    torch.distributed.all_gather_object(output_objects, object)
    if isinstance(object, (list, tuple)):
        output_list = []
        for item in output_objects:
            output_list.extend(item)
        return output_list
    elif isinstance(object, dict):
        template = output_objects[0]
        output_dict = {}
        for k, v in template.items():
            output_dict[k] = []
            for item in output_objects:
                if isinstance(item[k], list):
                    output_dict[k].extend(item[k])
                else:
                    output_dict[k].append(item[k])
        return output_dict


def gather_object(object: Any):
    """
    Recursively gather object in a nested list/tuple/dictionary of objects from all devices.

    Args:
        object (nested list/tuple/dictionary of picklable object):
            The data to gather.

    Returns:
        The same data structure as `object` with all the objects sent to every device.
    """
    if "tpu" in str(PartialState().distributed_type).lower():
        raise NotImplementedError("gather objects in TPU is not supported")
    elif PartialState().distributed_type in TORCH_DISTRIBUTED_OPERATION_TYPES:
        return _gpu_gather_object(object)
    else:
        return object


def gather_for_metrics(accelerator, input_data):
    """
    by JY Huang: re-implement this method for gathering non-tensor objects
    Refer source code to https://huggingface.co/docs/accelerate/package_reference/accelerator#accelerate.Accelerator.gather_for_metrics
    """

    try:
        recursively_apply(lambda x: x, input_data, error_on_other_type=True)
        all_tensors = True
    except TypeError:
        all_tensors = False

    if not all_tensors:
        data = gather_object(input_data)
    else:
        data = accelerator.gather(input_data)

    try:
        if accelerator.gradient_state.end_of_dataloader:
            # at the end of a dataloader, `gather_for_metrics` regresses to
            # `gather` unless the dataset has a remainder so log.
            if accelerator.gradient_state.remainder == -1:
                logger.info(
                    "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                )
                return data
            elif accelerator.gradient_state.remainder > 0:
                # Last batch needs to be truncated on distributed systems as it contains additional samples
                def _adjust_samples(tensor):
                    return tensor[: accelerator.gradient_state.remainder] if tensor is not None else None
                if all_tensors:
                    # This only applies to tensors, as defined in `recursively_apply`
                    return recursively_apply(_adjust_samples, data)
                else:
                    if isinstance(data, (list, tuple)):
                        return _adjust_samples(data)
                    elif isinstance(data, dict):
                        return {k: _adjust_samples(v) for k, v in data.items()}
                    else:
                        raise NotImplementedError(f"Non-tensor gather only supports list, tuple or dict")
            else:  # remainder is 0
                # no remainder even though at end of dataloader, so nothing to do.
                return data
        else:
            # Not at the end of the dataloader, no need to adjust the tensors
            return data
    except Exception:
        # Dataset had no length or raised an error
        return data
    
def gather_dict(accelerator, data_dict):
    data_dict_non_tensor = {k : v for k, v in data_dict.items() if not isinstance(v, torch.Tensor)}
    data_dict_non_tensor = gather_for_metrics(accelerator, data_dict_non_tensor)
    data_dict = {k : v for k, v in data_dict.items() if isinstance(v, torch.Tensor)}
    data_dict = gather_for_metrics(accelerator, data_dict)
    data_dict.update(data_dict_non_tensor)
    return data_dict
