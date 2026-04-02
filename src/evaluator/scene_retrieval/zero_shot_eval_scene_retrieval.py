#!/usr/bin/env python3

"""Zero-shot caption-to-scene retrieval benchmark runner."""


import argparse
import json
from pathlib import Path
import re
import shutil
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from evaluator.common.multimodal_models import (
    DEFAULT_DFN_MODEL_ID,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SIGLIP_MODEL_NAME,
    BaseMultimodalModel,
    build_model,
    resolve_scan_filename,
)

_CTRL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")
DEFAULT_HF_REPO_ID = "MatchLab/ScenePoint"


def _sanitize_json_snippet(s: str) -> str:
    """Clean malformed JSON fragments before parsing."""
    s = _CTRL_CHARS.sub("", s)
    s = s.replace("“", "\"").replace("”", "\"").replace("’", "'")
    s = re.sub(r",\s*(\]|\})", r"\1", s)
    s = re.sub(r"\bNaN\b", "null", s)
    s = re.sub(r"\bInfinity\b", "null", s)
    s = re.sub(r"\b-?Inf\b", "null", s)
    return s


def _split_or_accumulate_objects(raw_text: str):
    """Yield object-like chunks from noisy JSONL-style text."""
    s = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"}\s*{", "}\n{", s)
    lines = s.split("\n")
    buf = []
    depth = 0
    in_str = False
    esc = False

    def depth_of_chunk(chunk: str) -> int:
        nonlocal in_str, esc
        d = 0
        for ch in chunk:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch in "{[":
                    d += 1
                elif ch in "}]":
                    d -= 1
        return d

    for ln in lines:
        if not ln.strip():
            continue
        buf.append(ln)
        depth += depth_of_chunk(ln)
        if depth <= 0:
            yield "\n".join(buf)
            buf, depth, in_str, esc = [], 0, False, False

    if buf:
        yield "\n".join(buf)


def load_jsonl_group_by_scene(
    jsonl_path: str,
    n_utterances: int,
    joiner: str = " ",
    strategy: str = "first",
    min_len: int = 1,
    drop_last_incomplete: bool = True,
) -> List[Tuple[str, str]]:
    """Group caption rows by scene and merge them into retrieval queries."""
    with open(jsonl_path, "rb") as f:
        raw = f.read()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    text = raw.decode("utf-8", errors="replace")

    by_scene: Dict[str, List[str]] = {}
    bad_count = 0
    emitted = 0

    for i, chunk in enumerate(_split_or_accumulate_objects(text), 1):
        if not chunk.strip():
            continue
        s = _sanitize_json_snippet(chunk)
        try:
            obj = json.loads(s)
        except json.JSONDecodeError as e:
            bad_count += 1
            pos = e.pos
            preview = s[max(0, pos - 80): pos + 80]
            sys.stderr.write(
                f"[WARN] JSON object #{i} failed to parse: {e.msg} at pos {pos}\n"
                f"        preview: ...{preview}...\n"
            )
            continue

        sid = obj.get("scan_id")
        utt = (obj.get("utterance") or "").strip()
        if sid and len(utt) >= min_len:
            by_scene.setdefault(sid, []).append(utt)
            emitted += 1

    if bad_count:
        print(f"[WARN] Skipped {bad_count} malformed JSON objects; kept {emitted}.")

    scenes: List[Tuple[str, str]] = []
    for sid, utts in by_scene.items():
        if strategy == "random":
            g = torch.Generator().manual_seed(abs(hash(sid)) % (2**31))
            perm = torch.randperm(len(utts), generator=g).tolist()
            utts = [utts[i] for i in perm]
        chunk_size = max(1, int(n_utterances))
        for start in range(0, len(utts), chunk_size):
            chunk = utts[start:start + chunk_size]
            if drop_last_incomplete and len(chunk) < chunk_size:
                continue
            cap = joiner.join(chunk).strip()
            if cap:
                scenes.append((sid, cap))
    return scenes


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


def load_safetensor_from_hf(repo_id: str, filename: str, repo_type: str = "dataset") -> Dict[str, torch.Tensor]:
    """Download and load one safetensor file from Hugging Face."""
    cached_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type=repo_type,
        local_files_only=False,
    )
    return load_file(cached_path)


def load_safetensor(
    scan_id: str,
    scan_root: str,
    repo_id: str,
    filename_fmt: str,
    repo_type: str,
) -> Dict[str, torch.Tensor]:
    """Load one scene safetensor from local storage or Hugging Face."""
    filename = resolve_scan_filename(filename_fmt, scan_id)

    if scan_root:
        for local_path in _local_scan_candidates(scan_root, filename):
            if local_path.exists():
                return load_file(str(local_path))

    if repo_id:
        downloaded_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            local_files_only=False,
        )
        _cache_downloaded_scan(scan_root, filename, downloaded_path)
        return load_file(downloaded_path)

    raise FileNotFoundError(
        f"Could not locate scan '{scan_id}'. Provide a valid --scan_root or --hf_repo_id."
    )


def _load_single_scan_worker(
    sid: str,
    scan_root: str,
    repo_id: str,
    filename_fmt: str,
    repo_type: str,
    pm_key: str,
    rgb_key: str,
    max_views: Optional[int],
):
    """Load one scene inside a thread-pool worker."""
    try:
        sd = load_safetensor(sid, scan_root, repo_id, filename_fmt, repo_type)
        if (pm_key not in sd) or (rgb_key not in sd):
            return sid, "missing_keys", [k for k in (pm_key, rgb_key) if k not in sd]

        pm = sd[pm_key]
        rgb = sd[rgb_key]
        if max_views is not None:
            if hasattr(pm, "size") and pm.size(0) > max_views:
                pm = pm[:max_views]
            if rgb.size(0) > max_views:
                rgb = rgb[:max_views]
        return sid, "success", {"pointmaps": pm, "color_images": rgb}
    except Exception as e:
        return sid, "error", str(e)


def load_all_scene_pointmaps_from_hf(
    scan_ids: List[str],
    scan_root: str,
    repo_id: str,
    filename_fmt: str = "{scan_id}.safetensors",
    repo_type: str = "dataset",
    pm_key: str = "pointmaps",
    rgb_key: str = "color_images",
    max_views: Optional[int] = None,
    num_workers: int = 8,
) -> Dict[str, Dict[str, torch.Tensor]]:
    """Load scene tensors for all retrieval candidates."""
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    not_found = []
    missing_keys = []
    print(f"Starting multithreaded load for {len(scan_ids)} scans using {num_workers} workers...")

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_sid = {
            executor.submit(
                _load_single_scan_worker, sid, scan_root, repo_id, filename_fmt, repo_type, pm_key, rgb_key, max_views
            ): sid for sid in scan_ids
        }
        for future in tqdm(as_completed(future_to_sid), total=len(scan_ids)):
            sid = future_to_sid[future]
            try:
                status_sid, status, data = future.result()
                if status == "success":
                    out[status_sid] = data
                elif status == "missing_keys":
                    missing_keys.append((status_sid, data))
                else:
                    not_found.append((status_sid, data))
            except Exception as e:
                not_found.append((sid, str(e)))

    if not_found:
        print(f"[WARN] Failed/Missing {len(not_found)} scenes. Example: {not_found[:3]}")
    if missing_keys:
        print(f"[WARN] {len(missing_keys)} scenes missing keys. Example: {missing_keys[:3]}")
    return out


class SceneCaptionRetrievalDataset(Dataset):
    """Dataset wrapper that pairs caption queries with scene ids."""

    def __init__(
        self,
        model: BaseMultimodalModel,
        items: List[Tuple[str, str]],
        scene_pointmaps: Dict[str, Dict[str, torch.Tensor]],
        input_mode: str = "pm+image",
    ):
        """Precompute one embedding per candidate scene."""
        have = set(scene_pointmaps.keys())
        dropped = [(sid, cap) for sid, cap in items if sid not in have]
        if dropped:
            cnt = Counter([sid for sid, _ in dropped])
            print(f"[WARN] Dropped {len(dropped)} caption rows (no point maps). Top: {cnt.most_common(5)}")
        self.items = [(sid, cap) for sid, cap in items if sid in have]

        self.scene_ids = sorted(set(sid for sid, _ in self.items))
        self.sid_to_idx = {sid: i for i, sid in enumerate(self.scene_ids)}

        feats = []
        with torch.no_grad():
            for sid in self.scene_ids:
                pm = scene_pointmaps[sid]["pointmaps"]
                images = scene_pointmaps[sid]["color_images"]
                f = model.encode_views(images, pm, input_mode=input_mode)
                f = torch.mean(f, dim=0) if f.dim() == 2 else f
                feats.append(F.normalize(f.squeeze(0).float(), dim=-1).unsqueeze(0))

        if not feats:
            raise RuntimeError(
                "No scene features were built. This usually means no scan tensors were loaded. "
                "Check --scan_root, --filename_fmt, --hf_repo_id, and the tensor keys."
            )
        self.scene_feats = torch.cat(feats, dim=0)

    def __len__(self) -> int:
        """Return the number of caption queries."""
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return one caption query and its target scene index."""
        sid, caption = self.items[idx]
        return {"scene_id": sid, "caption": caption, "target_index": self.sid_to_idx[sid]}


def collate_simple(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate caption queries into a simple batch dictionary."""
    return {
        "scene_ids": [b["scene_id"] for b in batch],
        "captions": [b["caption"] for b in batch],
        "targets": torch.tensor([b["target_index"] for b in batch], dtype=torch.long),
    }


@torch.no_grad()
def retrieve_and_rank(text_embs: torch.Tensor, scene_feats: torch.Tensor) -> torch.Tensor:
    """Compute caption-to-scene similarity scores."""
    return F.normalize(text_embs, dim=-1) @ F.normalize(scene_feats, dim=-1).T


@torch.no_grad()
def ranks_and_metrics(sims: torch.Tensor, targets: torch.Tensor, ks=(1, 5, 10)) -> Dict[str, Any]:
    """Convert similarity scores into retrieval ranks and metrics."""
    order = torch.argsort(sims, dim=1, descending=True)
    match_positions = (order == targets.unsqueeze(1)).nonzero(as_tuple=False)
    match_positions = match_positions[torch.argsort(match_positions[:, 0])]
    ranks = match_positions[:, 1]
    metrics = {"MRR": float((1.0 / (ranks + 1).float()).mean().item()), "mean_rank": float(ranks.float().mean().item())}
    for k in ks:
        metrics[f"R@{k}"] = float((ranks < k).float().mean().item())
    return {"ranks": ranks.cpu().tolist(), "metrics": metrics, "order": order}


def main():
    """Run scene retrieval from the command line."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--scan_root", default="")
    parser.add_argument("--hf_repo_id", default=DEFAULT_HF_REPO_ID)
    parser.add_argument("--filename_fmt", default="{scan_id}.safetensors")
    parser.add_argument("--hf_repo_type", default="dataset", choices=["dataset", "model", "space"])
    parser.add_argument("--pm_key", default="pointmaps")
    parser.add_argument("--rgb_key", default="color_images")
    parser.add_argument("--max_views", type=int, default=None)
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--model_type", type=str, default="uniscene3d",
                        choices=("uniscene3d", "fgclip", "poma3d", "dfn", "siglip"))
    parser.add_argument("--input_mode", type=str, default="pm+image",
                        choices=("pm", "image", "pm+image"))
    parser.add_argument("--model_root", type=str, default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--dfn_model_name", type=str, default=DEFAULT_DFN_MODEL_ID)
    parser.add_argument("--siglip_model_name", type=str, default=DEFAULT_SIGLIP_MODEL_NAME)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--n_utterances", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--strategy", type=str, default="first", choices=["first", "random", "all"])
    parser.add_argument("--drop_last_incomplete", action="store_true")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"
    device = torch.device(args.device)

    model = build_model(args, config_text_max_length=248, config_walk_short_pos=False, config_siglip_text_max_length=64)

    items = load_jsonl_group_by_scene(
        jsonl_path=args.jsonl,
        n_utterances=args.n_utterances,
        strategy=args.strategy,
        drop_last_incomplete=args.drop_last_incomplete,
    )
    print("Evaluating all items in the JSONL.")
    if not args.scan_root and not args.hf_repo_id:
        raise ValueError("Provide at least one of --scan_root or --hf_repo_id.")
    scan_ids = sorted(set(s for s, _ in items))
    print(f"Built {len(items)} caption queries total (multi-captions per scene).")

    scene_pointmaps = load_all_scene_pointmaps_from_hf(
        scan_ids=scan_ids,
        scan_root=args.scan_root,
        repo_id=args.hf_repo_id,
        filename_fmt=args.filename_fmt,
        repo_type=args.hf_repo_type,
        pm_key=args.pm_key,
        rgb_key=args.rgb_key,
        max_views=args.max_views,
    )
    print(f"Loaded point maps for {len(scene_pointmaps)}/{len(scan_ids)} scenes from configured sources.")

    ds = SceneCaptionRetrievalDataset(model, items=items, scene_pointmaps=scene_pointmaps, input_mode=args.input_mode)
    print(f"Scene bank built: {len(ds.scene_ids)} scenes; feat_dim={ds.scene_feats.shape[-1]}")
    print(f"Caption queries kept: {len(ds)}")

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_simple,
        drop_last=False,
    )

    feat_bank = ds.scene_feats.to(device)
    agg = {"R@1": [], "R@5": [], "R@10": [], "MRR": []}

    with torch.no_grad():
        for batch in loader:
            caps = batch["captions"]
            tgt = batch["targets"].to(device)
            text_embs = model.encode_text(caps)
            sims = retrieve_and_rank(text_embs, feat_bank)
            out = ranks_and_metrics(sims, tgt, ks=(1, 5, 10))
            for k, v in out["metrics"].items():
                if k in agg:
                    agg[k].append(v)

    final = {k: (sum(v) / len(v) if v else 0.0) for k, v in agg.items()}
    print("== Retrieval (caption -> scene) ==")
    for k in ("R@1", "R@5", "R@10", "MRR"):
        print(f"{k}: {final[k]:.4f}")


if __name__ == "__main__":
    main()
