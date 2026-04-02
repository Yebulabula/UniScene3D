#!/usr/bin/env python3

"""Zero-shot caption-to-view retrieval benchmark runner."""


import argparse
import json
from pathlib import Path
import shutil
from typing import Dict, List, Tuple

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from safetensors.torch import load_file
from tqdm import tqdm

from evaluator.common.multimodal_models import (
    DEFAULT_DFN_MODEL_ID,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SIGLIP_MODEL_NAME,
    BaseMultimodalModel,
    build_model,
    resolve_scan_filename,
)

DEFAULT_HF_REPO_ID = "MatchLab/ScenePoint"


def load_jsonl(path: str) -> List[dict]:
    """Load retrieval queries from a JSONL file."""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def _local_scan_candidates(scan_root: str, filename: str) -> List[Path]:
    """Return the local file locations to try for one scan."""
    root = Path(scan_root)
    basename = Path(filename).name
    candidates = [
        root / filename,
        root / basename,
        root / "light_scannet" / basename,
    ]
    deduped = []
    seen = set()
    for candidate in candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _preferred_cache_path(scan_root: str, filename: str) -> Path:
    """Choose where a downloaded scan should be cached locally."""
    root = Path(scan_root)
    formatted = Path(filename)
    if len(formatted.parts) > 1 and root.name == formatted.parts[0]:
        return root / formatted.name
    return root / formatted


def _cache_downloaded_scan(scan_root: str, filename: str, downloaded_path: str) -> None:
    """Cache a downloaded scan file under the local scan root."""
    if not scan_root:
        return
    destination = _preferred_cache_path(scan_root, filename)
    if destination.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(downloaded_path, destination)


def load_scan_modalities(
    scan_id: str,
    scan_root: str,
    hf_repo_id: str,
    filename_fmt: str,
    repo_type: str,
) -> Dict[str, torch.Tensor]:
    """Load one scan from local storage or Hugging Face."""
    formatted = resolve_scan_filename(filename_fmt, scan_id)

    if scan_root:
        for local_path in _local_scan_candidates(scan_root, formatted):
            if local_path.exists():
                return load_file(str(local_path))

    if hf_repo_id:
        try:
            remote_path = hf_hub_download(
                repo_id=hf_repo_id,
                filename=formatted,
                repo_type=repo_type,
                local_files_only=False,
            )
            _cache_downloaded_scan(scan_root, formatted, remote_path)
        except EntryNotFoundError as exc:
            raise FileNotFoundError(
                f"HF file not found for scan '{scan_id}'. "
                f"Resolved filename='{formatted}', repo='{hf_repo_id}', repo_type='{repo_type}'."
            ) from exc
        return load_file(remote_path)

    raise FileNotFoundError(
        f"Could not locate scan '{scan_id}'. Provide a valid --scan_root or --hf_repo_id."
    )


@torch.no_grad()
def encode_scan_views(
    model: BaseMultimodalModel,
    images: torch.Tensor,
    point_maps: torch.Tensor,
    input_mode: str,
    batch_views: int,
) -> torch.Tensor:
    """Encode all views for one scan in smaller batches."""
    outputs = []
    total_views = images.shape[0]
    for start in range(0, total_views, batch_views):
        end = min(start + batch_views, total_views)
        outputs.append(
            model.encode_views(images[start:end], point_maps[start:end], input_mode=input_mode).detach().cpu()
        )
    return torch.cat(outputs, dim=0)


@torch.no_grad()
def eval_view_retrieval(
    model: BaseMultimodalModel,
    items: List[dict],
    scan_root: str,
    hf_repo_id: str,
    filename_fmt: str,
    repo_type: str,
    pm_key: str,
    rgb_key: str,
    input_mode: str,
    batch_views: int = 32,
    recall_ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    """Evaluate caption-to-view retrieval for a list of queries."""
    scan_cache: Dict[str, torch.Tensor] = {}
    total = 0
    top1_correct = 0
    visible_top1_correct = 0
    recall_correct = {k: 0 for k in recall_ks}
    visible_recall_correct = {k: 0 for k in recall_ks}

    for item in tqdm(items, desc="View retrieval", unit="query"):
        scan_id = item["scan_id"]
        utterance = item["utterance"]
        gt_views = item.get("view_ground_truth")
        if not gt_views:
            continue

        gt = int(gt_views[0])
        visible_views = {int(v) for v in gt_views}

        if scan_id not in scan_cache:
            scan_data = load_scan_modalities(scan_id, scan_root, hf_repo_id, filename_fmt, repo_type)
            scan_cache[scan_id] = encode_scan_views(
                model=model,
                images=scan_data[rgb_key],
                point_maps=scan_data[pm_key],
                input_mode=input_mode,
                batch_views=batch_views,
            )

        view_feats = scan_cache[scan_id]
        num_views = view_feats.shape[0]
        visible_views = {v for v in visible_views if 0 <= v < num_views}
        if gt < 0 or gt >= num_views or not visible_views:
            continue

        text_feat = model.encode_text([utterance]).squeeze(0).detach().cpu()
        sims = view_feats @ text_feat
        ranked = torch.argsort(sims, descending=True)
        pred = int(ranked[0].item())

        total += 1
        if pred == gt:
            top1_correct += 1
        if pred in visible_views:
            visible_top1_correct += 1

        for k in recall_ks:
            topk = ranked[: min(k, num_views)].tolist()
            if gt in topk:
                recall_correct[k] += 1
            if any(v in visible_views for v in topk):
                visible_recall_correct[k] += 1

    if total == 0:
        return {"n": 0}

    metrics: Dict[str, float] = {
        "n": total,
        "top1_acc": top1_correct / total,
        "visible_view_acc": visible_top1_correct / total,
    }
    for k in recall_ks:
        metrics[f"recall@{k}"] = recall_correct[k] / total
        metrics[f"visible_recall@{k}"] = visible_recall_correct[k] / total
    return metrics


def main() -> None:
    """Run view retrieval from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True)
    parser.add_argument("--scan_root", type=str, default="")
    parser.add_argument("--hf_repo_id", type=str, default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--filename_fmt", type=str, default="{scan_id}.safetensors")
    parser.add_argument("--repo_type", type=str, default="dataset")
    parser.add_argument("--pm_key", type=str, default="point_map")
    parser.add_argument("--rgb_key", type=str, default="color_images")
    parser.add_argument("--ckpt", type=str, default="")
    parser.add_argument("--model_root", type=str, default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--model_type", type=str, default="uniscene3d",
                        choices=("uniscene3d", "fgclip", "poma3d", "siglip", "dfn"))
    parser.add_argument("--input_mode", type=str, default="pm+image",
                        choices=("pm", "image", "pm+image"))
    parser.add_argument("--dfn_model_name", type=str, default=DEFAULT_DFN_MODEL_ID)
    parser.add_argument("--siglip_model_name", type=str, default=DEFAULT_SIGLIP_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_views", type=int, default=32)
    parser.add_argument("--max_items", type=int, default=-1)
    args = parser.parse_args()

    items = load_jsonl(args.jsonl)
    print("Evaluating all items in the JSONL.")
    # if args.max_items > 0:
    items = items

    if not args.scan_root and not args.hf_repo_id:
        raise ValueError("Provide at least one of --scan_root or --hf_repo_id.")

    model = build_model(args, config_text_max_length=77, config_walk_short_pos=True, config_siglip_text_max_length=64)
    metrics = eval_view_retrieval(
        model=model,
        items=items,
        scan_root=args.scan_root,
        hf_repo_id=args.hf_repo_id,
        filename_fmt=args.filename_fmt,
        repo_type=args.repo_type,
        pm_key=args.pm_key,
        rgb_key=args.rgb_key,
        input_mode=args.input_mode,
        batch_views=args.batch_views,
        recall_ks=(1, 5, 10),
    )

    print("\n=== View Retrieval Results ===")
    print(f"model_type: {args.model_type}")
    print(f"input_mode: {args.input_mode}")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:>18}: {value:.4f}")
        else:
            print(f"{key:>18}: {value}")


if __name__ == "__main__":
    main()
