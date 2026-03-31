#!/usr/bin/env python3

"""Few-shot scene classification benchmark runner."""


import argparse
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from evaluator.scene_classification.zero_shot_scene_cls import (
    DEFAULT_DFN_MODEL_ID,
    DEFAULT_MODEL_ROOT,
    DEFAULT_SIGLIP_MODEL_NAME,
    build_model,
    load_all_scene_modalities_from_hf,
    load_room_type_json,
    resize_to_224_if_needed,
    to_vchw,
)


def stratified_fewshot_train_val_test(
    scan_ids: List[str],
    labels: List[int],
    n_train_per_class: int,
    n_val_per_class: int,
    seed: int,
) -> Tuple[List[int], List[int], List[int]]:
    """Build per-class train, val, and test splits."""
    g = torch.Generator().manual_seed(seed)

    idx_by_class: Dict[int, List[int]] = defaultdict(list)
    for i, y in enumerate(labels):
        idx_by_class[y].append(i)

    train_idx, val_idx = [], []
    for y, idxs in idx_by_class.items():
        perm = torch.randperm(len(idxs), generator=g).tolist()
        idxs = [idxs[p] for p in perm]

        t = min(n_train_per_class, len(idxs))
        train_idx.extend(idxs[:t])
        rest = idxs[t:]

        v = min(n_val_per_class, len(rest))
        val_idx.extend(rest[:v])

    used = set(train_idx) | set(val_idx)
    test_idx = [i for i in range(len(scan_ids)) if i not in used]
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


@torch.no_grad()
def compute_scene_embedding(model, rgb: torch.Tensor, pm: torch.Tensor, input_mode: str) -> torch.Tensor:
    """Average view features into one scene embedding."""
    vfeats = model.encode_views(rgb, pm, input_mode=input_mode)
    sfeat = vfeats.mean(dim=0)
    return F.normalize(sfeat.float(), dim=-1)


def clip_style_logreg_lbfgs(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    lambdas: np.ndarray,
    max_iter: int = 1000,
) -> Tuple[LogisticRegression, float, float]:
    """Tune a multinomial logistic regressor over lambda values."""
    best_acc = -1.0
    best_lambda = None
    best_clf = None

    if X_val.shape[0] == 0:
        lam = 1.0
        clf = LogisticRegression(
            penalty="l2",
            C=1.0 / lam,
            solver="lbfgs",
            max_iter=max_iter,
            multi_class="multinomial",
            n_jobs=None,
        )
        clf.fit(X_train, y_train)
        return clf, lam, float("nan")

    for lam in lambdas:
        clf = LogisticRegression(
            penalty="l2",
            C=1.0 / float(lam),
            solver="lbfgs",
            max_iter=max_iter,
            multi_class="multinomial",
            n_jobs=None,
        )
        clf.fit(X_train, y_train)
        acc = accuracy_score(y_val, clf.predict(X_val))
        if acc > best_acc:
            best_acc = acc
            best_lambda = float(lam)
            best_clf = clf

    assert best_clf is not None
    return best_clf, best_lambda, best_acc


def main() -> None:
    """Run few-shot scene classification from the command line."""
    parser = argparse.ArgumentParser("Few-shot room-type classification via linear probe")
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
    parser.add_argument("--n_train_per_class", type=int, default=1)
    parser.add_argument("--n_val_per_class", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--normalize_labels", action="store_true")
    parser.add_argument("--lambda_min", type=float, default=1e-6)
    parser.add_argument("--lambda_max", type=float, default=1e6)
    parser.add_argument("--lambda_steps", type=int, default=96)
    parser.add_argument("--max_iter", type=int, default=1000)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable. Falling back to CPU.")
        args.device = "cpu"

    sid_to_room = load_room_type_json(args.room_type_json)
    if args.normalize_labels:
        sid_to_room = {k: v.lower().replace("_", " ").replace("/", "or").strip() for k, v in sid_to_room.items()}

    scan_ids_all = sorted(sid_to_room.keys())
    room_types = [sid_to_room[sid] for sid in scan_ids_all]
    classes = sorted(set(room_types))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    y_all = [class_to_idx[r] for r in room_types]

    print(f"Loaded labels: scenes={len(scan_ids_all)} classes={len(classes)}")
    print("Class counts:", dict(Counter(room_types)))

    scene_data = load_all_scene_modalities_from_hf(
        scan_ids=scan_ids_all,
        repo_id=args.hf_repo_id,
        filename_fmt=args.filename_fmt,
        repo_type=args.hf_repo_type,
        pm_key=args.pm_key,
        rgb_key=args.rgb_key,
    )
    scan_ids = [sid for sid in scan_ids_all if sid in scene_data]
    if len(scan_ids) != len(scan_ids_all):
        print(f"[WARN] Dropped {len(scan_ids_all) - len(scan_ids)} scenes due to missing HF data/keys.")
    y = [class_to_idx[sid_to_room[sid]] for sid in scan_ids]

    train_idx, val_idx, test_idx = stratified_fewshot_train_val_test(
        scan_ids=scan_ids,
        labels=y,
        n_train_per_class=args.n_train_per_class,
        n_val_per_class=args.n_val_per_class,
        seed=args.seed,
    )
    print(
        f"Split: train={len(train_idx)} val={len(val_idx)} test={len(test_idx)} "
        f"(train_shots={args.n_train_per_class}, val_shots={args.n_val_per_class})"
    )

    model = build_model(
        args,
        config_text_max_length=77,
        config_walk_short_pos=True,
        config_siglip_text_max_length=64,
    )

    feats = []
    for sid in scan_ids:
        pm = resize_to_224_if_needed(to_vchw(scene_data[sid]["pointmaps"]))
        rgb = scene_data[sid]["color_images"]
        feats.append(compute_scene_embedding(model, rgb, pm, input_mode=args.input_mode).unsqueeze(0))

    X_all = torch.cat(feats, dim=0)
    y_all_t = torch.tensor(y, dtype=torch.long)

    X_np = X_all.detach().cpu().numpy().astype(np.float32)
    y_np = y_all_t.numpy().astype(np.int64)

    X_train, y_train = X_np[train_idx], y_np[train_idx]
    X_val, y_val = X_np[val_idx], y_np[val_idx]
    X_test, y_test = X_np[test_idx], y_np[test_idx]

    lambdas = np.logspace(np.log10(args.lambda_min), np.log10(args.lambda_max), args.lambda_steps)
    clf, best_lambda, val_acc = clip_style_logreg_lbfgs(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        lambdas=lambdas,
        max_iter=args.max_iter,
    )

    train_acc = accuracy_score(y_train, clf.predict(X_train)) if X_train.shape[0] else float("nan")
    test_acc = accuracy_score(y_test, clf.predict(X_test)) if X_test.shape[0] else float("nan")

    print("== Few-shot Linear Probe (Original CLIP style) ==")
    print(f"Best lambda: {best_lambda:.6g} | Val Acc: {val_acc if not np.isnan(val_acc) else 'N/A'}")
    print(f"Train Acc: {train_acc:.4f}  (N={len(train_idx)})")
    print(f"Test  Acc: {test_acc:.4f}  (N={len(test_idx)})")


if __name__ == "__main__":
    main()
