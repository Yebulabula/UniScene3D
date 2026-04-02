#!/usr/bin/env python3

"""Zero-shot scene classification benchmark runner."""


import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from tqdm import tqdm

from evaluator.common.multimodal_models import (
    DEFAULT_DFN_MODEL_ID,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SIGLIP_MODEL_NAME,
    BaseMultimodalModel,
    build_model,
    resolve_scan_filename,
    resize_to_224_if_needed,
    to_vchw,
)


def load_safetensor_from_hf(repo_id: str, filename: str, repo_type: str = "dataset") -> Dict[str, torch.Tensor]:
    """Download and load one safetensor file from Hugging Face."""
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_files_only=False,
    )
    return load_file(cached_path)


def process_single_scan(sid, repo_id, filename_fmt, repo_type, pm_key, rgb_key):
    """Load one scene's RGB and point-map tensors."""
    try:
        filename = resolve_scan_filename(filename_fmt, sid)
        sd = load_safetensor_from_hf(repo_id, filename, repo_type=repo_type)
        return sid, {"pointmaps": sd[pm_key], "color_images": sd[rgb_key]}
    except Exception as e:
        print(f"Error processing {sid}: {e}")
        return sid, None


def load_all_scene_modalities_from_hf(
    scan_ids: List[str],
    repo_id: str,
    filename_fmt: str = "{scan_id}.safetensors",
    repo_type: str = "dataset",
    pm_key: str = "pointmaps",
    rgb_key: str = "color_images",
    max_workers: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load RGB and point-map tensors for a list of scenes."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    worker = partial(
        process_single_scan,
        repo_id=repo_id,
        filename_fmt=filename_fmt,
        repo_type=repo_type,
        pm_key=pm_key,
        rgb_key=rgb_key,
    )

    print(f"Starting parallel load with {max_workers} workers...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(executor.map(worker, scan_ids), total=len(scan_ids)))

    for sid, data in results:
        if data is not None:
            out[sid] = data
    return out


def load_room_type_json(path: str) -> Dict[str, str]:
    """Parse a room-type annotation file into a scan-to-label map."""
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    mapping: Dict[str, str] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            if isinstance(v, str):
                mapping[k] = v
            elif isinstance(v, dict):
                if "room_type" in v and isinstance(v["room_type"], str):
                    mapping[k] = v["room_type"]
                elif "label" in v and isinstance(v["label"], str):
                    mapping[k] = v["label"]
    elif isinstance(obj, list):
        for it in obj:
            if not isinstance(it, dict):
                continue
            sid = it.get("scan_id") or it.get("scene_id")
            rt = it.get("room_type") or it.get("label") or it.get("room")
            if isinstance(sid, str) and isinstance(rt, str):
                mapping[sid] = rt

    if not mapping:
        raise ValueError(f"Could not parse any (scan_id -> room_type) mapping from: {path}")
    return mapping


def build_prompts_for_classes(classes: List[str], templates: List[str]) -> Tuple[List[str], List[int]]:
    """Expand class labels into text prompts."""
    prompts, prompt_to_class = [], []
    for ci, room_type in enumerate(classes):
        for template in templates:
            prompts.append(template.format(room_type=room_type))
            prompt_to_class.append(ci)
    return prompts, prompt_to_class


@torch.no_grad()
def zero_shot_classify(
    model: BaseMultimodalModel,
    scene_embs: torch.Tensor,
    class_text_embs: torch.Tensor,
    prompt_to_class: List[int],
    num_classes: int,
) -> torch.Tensor:
    """Predict the best class for each scene embedding."""
    del model
    scene_embs = F.normalize(scene_embs, dim=-1)
    class_text_embs = F.normalize(class_text_embs, dim=-1)

    sims = scene_embs @ class_text_embs.t()
    scores = torch.full((scene_embs.size(0), num_classes), -1e9, device=sims.device, dtype=sims.dtype)
    p2c = torch.tensor(prompt_to_class, device=sims.device, dtype=torch.long)
    scores.scatter_reduce_(dim=1, index=p2c.unsqueeze(0).expand(sims.size(0), -1), src=sims, reduce="amax")
    return scores.argmax(dim=1)


def main():
    """Run zero-shot scene classification from the command line."""
    parser = argparse.ArgumentParser("Zero-shot room-type classification (accuracy only)")
    parser.add_argument("--room_type_json", required=True)
    parser.add_argument("--hf_repo_id", required=True)
    parser.add_argument("--filename_fmt", default="{scan_id}.safetensors")
    parser.add_argument("--hf_repo_type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--pm_key", default="pointmaps")
    parser.add_argument("--rgb_key", default="color_images")
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--model_type", type=str, default="uniscene3d",
                        choices=("uniscene3d", "fgclip", "poma3d", "dfn", "siglip"))
    parser.add_argument("--input_mode", type=str, default="pm+image",
                        choices=("pm", "image", "pm+image"))
    parser.add_argument("--dfn_model_name", type=str, default=DEFAULT_DFN_MODEL_ID)
    parser.add_argument("--siglip_model_name", type=str, default=DEFAULT_SIGLIP_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_scenes", type=int, default=8)
    parser.add_argument("--templates", nargs="*", default=["this room is a {room_type}."])
    parser.add_argument("--normalize_labels", action="store_true")
    parser.add_argument("--print_confusion_topk", type=int, default=0)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    sid_to_room = load_room_type_json(args.room_type_json)
    if args.normalize_labels:
        sid_to_room = {k: v.lower().replace("/", "or").strip() for k, v in sid_to_room.items()}

    scan_ids = sorted(sid_to_room.keys())
    print(f"Loaded labels for {len(scan_ids)} scenes from {args.room_type_json}")

    classes = sorted(set(sid_to_room.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_true = torch.tensor([class_to_idx[sid_to_room[sid]] for sid in scan_ids], dtype=torch.long)

    scene_data = load_all_scene_modalities_from_hf(
        scan_ids=scan_ids,
        repo_id=args.hf_repo_id,
        filename_fmt=args.filename_fmt,
        repo_type=args.hf_repo_type,
        pm_key=args.pm_key,
        rgb_key=args.rgb_key,
    )
    kept_scan_ids = [sid for sid in scan_ids if sid in scene_data]
    if len(kept_scan_ids) != len(scan_ids):
        print(f"[WARN] Dropped {len(scan_ids) - len(kept_scan_ids)} scenes (missing HF data/keys).")
    y_true = torch.tensor([class_to_idx[sid_to_room[sid]] for sid in kept_scan_ids], dtype=torch.long)

    model = build_model(
        args,
        config_text_max_length=77,
        config_walk_short_pos=True,
        config_siglip_text_max_length=64,
    )

    templates = [t.lower() for t in args.templates] if args.normalize_labels else args.templates
    prompts, prompt_to_class = build_prompts_for_classes(classes, templates)
    print(f"Text prompts: {len(prompts)} (classes={len(classes)} x templates={len(templates)})")
    class_text_embs = model.encode_text(prompts)

    scene_emb_list: List[torch.Tensor] = []
    bs = max(1, int(args.batch_scenes))
    for start in tqdm(range(0, len(kept_scan_ids), bs), desc="Encoding scenes", unit="batch"):
        for sid in kept_scan_ids[start:start + bs]:
            pm = to_vchw(scene_data[sid]["pointmaps"])
            rgb = scene_data[sid]["color_images"]
            vfeats = model.encode_views(rgb, pm, input_mode=args.input_mode)
            sfeats = F.normalize(vfeats.mean(dim=0, keepdim=True).float(), dim=-1)
            scene_emb_list.append(sfeats)

    scene_embs = torch.cat(scene_emb_list, dim=0)
    print(f"Scene embeddings: {scene_embs.shape}")

    y_pred = zero_shot_classify(
        model=model,
        scene_embs=scene_embs,
        class_text_embs=class_text_embs,
        prompt_to_class=prompt_to_class,
        num_classes=len(classes),
    ).cpu()

    correct = (y_pred == y_true).sum().item()
    total = y_true.numel()
    acc = correct / total if total else 0.0
    print("== Zero-shot Room-Type Classification ==")
    print(f"Accuracy: {acc:.4f}  ({correct}/{total})")


if __name__ == "__main__":
    main()
