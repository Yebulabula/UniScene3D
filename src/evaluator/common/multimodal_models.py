"""Shared model wrappers used by benchmark evaluators."""

import glob
import os
from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from open_clip import create_model_from_pretrained, get_tokenizer
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from transformers import (
    AutoImageProcessor,
    AutoModel,
    AutoProcessor,
    AutoTokenizer,
)

from common.misc import build_fgclip_model_from_local_code_with_hf_weights
from evaluator.common.paths import DEFAULT_MODEL_ROOT

DEFAULT_DFN_MODEL_ID = "hf-hub:apple/DFN2B-CLIP-ViT-B-16"
DEFAULT_SIGLIP_MODEL_NAME = "google/siglip2-base-patch16-224"


@dataclass
class ModelBuildConfig:
    """Shared text settings for evaluator model wrappers."""

    text_max_length: int
    walk_short_pos: bool
    siglip_text_max_length: int = 64


def resolve_scan_filename(filename_fmt: str, scan_id: str) -> str:
    """Resolve a scan id into a safetensor filename."""
    # Accept either printf-style, template-style, or literal safetensor paths.
    if "{scan_id}" in filename_fmt:
        return filename_fmt.replace("{scan_id}", scan_id)
    if "%s" in filename_fmt:
        return filename_fmt % scan_id
    if filename_fmt.endswith(".safetensors"):
        return filename_fmt
    return f"{filename_fmt.rstrip('/')}/{scan_id}.safetensors"


def extract_feature_tensor(output: Any) -> torch.Tensor:
    """Extract the last tensor feature from a backbone output."""
    # Different backbones return either tensors directly or tuples of tensors.
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        for item in reversed(output):
            if isinstance(item, torch.Tensor):
                return item
    raise TypeError(f"Unsupported feature output type: {type(output)!r}")


def to_vchw(tensor: torch.Tensor) -> torch.Tensor:
    """Convert image-like tensors to (V, C, H, W)."""
    if tensor.dim() != 4:
        raise ValueError(f"Expected a 4D tensor, got {tuple(tensor.shape)}")
    if tensor.shape[1] == 3:
        return tensor.float()
    if tensor.shape[-1] == 3:
        return tensor.permute(0, 3, 1, 2).contiguous().float()
    raise ValueError(f"Cannot convert tensor to (V,3,H,W): {tuple(tensor.shape)}")


def resize_to_224_if_needed(tensor: torch.Tensor) -> torch.Tensor:
    """Resize view tensors to the backbone input size."""
    if tensor.shape[-2:] != (224, 224):
        tensor = F.interpolate(tensor, size=(224, 224), mode="bilinear", align_corners=False)
    return tensor


def load_pretrain(model: nn.Module, pretrain_ckpt_path: str):
    """Load evaluator checkpoint weights into a wrapped model."""
    print(f"Loading pretrained weights from: {str(pretrain_ckpt_path)}")

    weight_files: List[str] = []
    if os.path.isdir(pretrain_ckpt_path):
        weight_files = sorted(glob.glob(os.path.join(pretrain_ckpt_path, "model*.safetensors")))
    elif os.path.isfile(pretrain_ckpt_path):
        if pretrain_ckpt_path.endswith(".safetensors"):
            weight_files = [pretrain_ckpt_path]
        elif pretrain_ckpt_path.endswith((".pth", ".pt")):
            state = torch.load(pretrain_ckpt_path, map_location="cpu")
            # Strip common wrapper prefixes before loading old checkpoints.
            if isinstance(state, dict) and any(k.startswith(("model.", "target_model.")) for k in state.keys()):
                state = {
                    k.split(".", 1)[1] if k.startswith(("model.", "target_model.")) else k: v
                    for k, v in state.items()
                }
            print(f"Checkpoint keys ({len(state)}):")
            for key in sorted(state.keys()):
                print(key)
            result = model.load_state_dict(state, strict=False)
            print(f"Loaded .pth/.pt | missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")
            return
        else:
            raise FileNotFoundError(f"Unsupported checkpoint extension: {pretrain_ckpt_path}")
    else:
        raise FileNotFoundError(f"Path not found: {pretrain_ckpt_path}")

    if len(weight_files) == 0:
        raise FileNotFoundError(f"No model*.safetensors found in {pretrain_ckpt_path}")

    weights = {}
    for wf in weight_files:
        print(f"Loading weights from: {wf}")
        weights.update(load_file(wf, device="cpu"))

    result = model.load_state_dict(weights, strict=False)
    print(f"Loaded .safetensors | missing={len(result.missing_keys)} unexpected={len(result.unexpected_keys)}")


class BaseMultimodalModel(nn.Module):
    """Base wrapper for retrieval and classification backbones."""

    supported_input_modes: Tuple[str, ...] = ("image",)

    def __init__(self, device: str, config: ModelBuildConfig) -> None:
        """Store device and text settings for subclasses."""
        super().__init__()
        self.device = torch.device(device)
        self.config = config

    def validate_input_mode(self, input_mode: str) -> None:
        """Check whether the wrapper supports the requested modality mix."""
        if input_mode not in self.supported_input_modes:
            supported = ", ".join(self.supported_input_modes)
            raise ValueError(
                f"{self.__class__.__name__} does not support input_mode='{input_mode}'. "
                f"Supported modes: {supported}"
            )

    def encode_views(self, images: torch.Tensor, point_maps: torch.Tensor, input_mode: str) -> torch.Tensor:
        """Encode a batch of views into normalized features."""
        raise NotImplementedError

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode a list of texts into normalized features."""
        raise NotImplementedError


class UniScene3DModel(BaseMultimodalModel):
    """Evaluator wrapper for the UniScene3D backbone."""

    supported_input_modes = ("pm", "image", "pm+image")

    def __init__(self, model_root: str, device: str, config: ModelBuildConfig) -> None:
        """Load the UniScene3D backbone and preprocessing tools."""
        super().__init__(device, config)
        self.pm_encoder = build_fgclip_model_from_local_code_with_hf_weights(model_root)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_root, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_root, use_fast=True, local_files_only=True
        )

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the backbone image preprocessing pipeline."""
        # Reuse the backbone image processor so inputs match training-time normalization.
        processed = self.image_processor.preprocess(
            images,
            do_center_crop=False,
            do_resize=True,
            size={"height": 224, "width": 224},
            return_tensors="pt",
        )["pixel_values"]
        return processed.to(self.device, non_blocking=True)

    def _prepare_pointmaps(self, point_maps: torch.Tensor) -> torch.Tensor:
        """Resize and move point maps to the target device."""
        point_maps = resize_to_224_if_needed(to_vchw(point_maps))
        return point_maps.to(self.device, non_blocking=True)

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor, point_maps: torch.Tensor, input_mode: str) -> torch.Tensor:
        """Encode RGB, point-map, or fused view inputs."""
        self.validate_input_mode(input_mode)
        image_tensor = self._prepare_images(images)
        point_tensor = self._prepare_pointmaps(point_maps)
        if input_mode == "pm":
            model_input = point_tensor
        elif input_mode == "image":
            model_input = image_tensor
        else:
            model_input = torch.cat([image_tensor, point_tensor], dim=1)
        feats = extract_feature_tensor(self.pm_encoder.get_image_features(model_input))
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts with the UniScene3D text tower."""
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.text_max_length,
            return_tensors="pt",
        ).to(self.device)
        feats = self.pm_encoder.get_text_features(
            tok["input_ids"], walk_short_pos=self.config.walk_short_pos
        )
        return F.normalize(feats.float(), dim=-1)


class FGClipModel(UniScene3DModel):
    """Evaluator wrapper for RGB-only FG-CLIP usage."""

    supported_input_modes = ("image",)


class POMA3DModel(BaseMultimodalModel):
    """Evaluator wrapper for the POMA3D LoRA variant."""

    supported_input_modes = ("pm", "image", "pm+image")

    def __init__(self, model_root: str, device: str, config: ModelBuildConfig) -> None:
        """Load the base FG-CLIP model and attach LoRA adapters."""
        super().__init__(device, config)
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "k_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
            task_type="FEATURE_EXTRACTION",
        )
        base = build_fgclip_model_from_local_code_with_hf_weights(model_root)
        # POMA3D keeps the same backbone but adds lightweight LoRA adapters.
        self.target_model = get_peft_model(base, lora_config)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_root, trust_remote_code=True, use_fast=True, local_files_only=True
        )
        self.image_processor = AutoImageProcessor.from_pretrained(
            model_root, use_fast=True, local_files_only=True
        )

    def _prepare_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply the image preprocessing pipeline used by POMA3D."""
        processed = self.image_processor.preprocess(
            images,
            do_center_crop=False,
            do_resize=True,
            size={"height": 224, "width": 224},
            return_tensors="pt",
        )["pixel_values"]
        return processed.to(self.device, non_blocking=True)

    def _prepare_pointmaps(self, point_maps: torch.Tensor) -> torch.Tensor:
        """Resize and move point maps to the target device."""
        point_maps = resize_to_224_if_needed(to_vchw(point_maps))
        return point_maps.to(self.device, non_blocking=True)

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor, point_maps: torch.Tensor, input_mode: str) -> torch.Tensor:
        """Encode view inputs with the adapted POMA3D backbone."""
        self.validate_input_mode(input_mode)
        image_tensor = self._prepare_images(images)
        point_tensor = self._prepare_pointmaps(point_maps)
        if input_mode == "pm":
            model_input = point_tensor
        elif input_mode == "image":
            model_input = image_tensor
        else:
            model_input = torch.cat([image_tensor, point_tensor], dim=1)
        feats = extract_feature_tensor(self.target_model.get_image_features(model_input))
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts with the POMA3D text tower."""
        tok = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.text_max_length,
            return_tensors="pt",
        ).to(self.device)
        feats = self.target_model.get_text_features(
            tok["input_ids"], walk_short_pos=self.config.walk_short_pos
        )
        return F.normalize(feats.float(), dim=-1)


class DFNModel(BaseMultimodalModel):
    """Evaluator wrapper for the DFN CLIP backbone."""

    supported_input_modes = ("image",)

    def __init__(self, model_name: str, device: str, config: ModelBuildConfig) -> None:
        """Load the DFN model and normalization settings."""
        super().__init__(device, config)
        self.model, self.preprocess_full = create_model_from_pretrained(model_name)
        self.tokenizer = get_tokenizer("ViT-B-16")
        self.normalize = T.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        )
        self.model.eval()

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor, point_maps: torch.Tensor, input_mode: str) -> torch.Tensor:
        """Encode RGB views with the DFN image encoder."""
        del point_maps
        self.validate_input_mode(input_mode)
        image_tensor = resize_to_224_if_needed(to_vchw(images)).to(self.device, non_blocking=True).float()
        if image_tensor.max() > 1.0:
            image_tensor = image_tensor / 255.0
        image_tensor = self.normalize(image_tensor)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            feats = self.model.encode_image(image_tensor)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts with the DFN text encoder."""
        tok = self.tokenizer(texts, context_length=self.model.context_length).to(self.device)
        use_amp = self.device.type == "cuda"
        with torch.autocast(device_type=self.device.type, enabled=use_amp):
            feats = self.model.encode_text(tok)
        return F.normalize(feats.float(), dim=-1)


class SigLIPModel(BaseMultimodalModel):
    """Evaluator wrapper for the SigLIP backbone."""

    supported_input_modes = ("image",)

    def __init__(self, model_name: str, device: str, config: ModelBuildConfig) -> None:
        """Load the SigLIP model and processor."""
        super().__init__(device, config)
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

    @torch.no_grad()
    def encode_views(self, images: torch.Tensor, point_maps: torch.Tensor, input_mode: str) -> torch.Tensor:
        """Encode RGB views with the SigLIP image tower."""
        del point_maps
        self.validate_input_mode(input_mode)
        image_tensor = to_vchw(images).float().cpu()
        inputs = self.processor(
            images=[img for img in image_tensor], return_tensors="pt"
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        feats = self.model.get_image_features(**inputs)
        return F.normalize(feats.float(), dim=-1)

    @torch.no_grad()
    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """Encode text prompts with the SigLIP text tower."""
        inputs = self.processor(
            text=texts,
            padding="max_length",
            truncation=True,
            max_length=self.config.siglip_text_max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}
        feats = self.model.get_text_features(**inputs)
        return F.normalize(feats.float(), dim=-1)


def build_model(
    args: Any,
    config: Optional[ModelBuildConfig] = None,
    config_text_max_length: Optional[int] = None,
    config_walk_short_pos: Optional[bool] = None,
    config_siglip_text_max_length: int = 64,
) -> BaseMultimodalModel:
    """Build the evaluator backbone requested on the command line."""
    if config is None:
        if config_text_max_length is None or config_walk_short_pos is None:
            raise ValueError("Either config or both config_text_max_length/config_walk_short_pos must be provided.")
        config = ModelBuildConfig(
            text_max_length=config_text_max_length,
            walk_short_pos=config_walk_short_pos,
            siglip_text_max_length=config_siglip_text_max_length,
        )
    if args.model_type == "uniscene3d":
        model = UniScene3DModel(model_root=args.model_root, device=args.device, config=config)
    elif args.model_type == "fgclip":
        model = FGClipModel(model_root=args.model_root, device=args.device, config=config)
    elif args.model_type == "poma3d":
        model = POMA3DModel(model_root=args.model_root, device=args.device, config=config)
    elif args.model_type == "dfn":
        model = DFNModel(model_name=args.dfn_model_name, device=args.device, config=config)
    elif args.model_type == "siglip":
        model = SigLIPModel(model_name=args.siglip_model_name, device=args.device, config=config)
    else:
        raise ValueError(f"Unknown model_type: {args.model_type}")

    model = model.to(args.device).eval()
    if getattr(args, "ckpt", None):
        load_pretrain(model, args.ckpt)
    model.validate_input_mode(args.input_mode)
    return model
