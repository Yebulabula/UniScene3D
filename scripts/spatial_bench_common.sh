#!/usr/bin/env bash

set -euo pipefail

setup_spatial_bench_env() {
  local caller_dir="$1"
  PROJECT_ROOT="$(cd "${caller_dir}/../.." && pwd)"
  export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
  DATASET_ROOT="${PROJECT_ROOT}/dataset"
  RETRIEVAL_DIR="${DATASET_ROOT}/retrieval"
  CLASSIFICATION_DIR="${DATASET_ROOT}/classification"
  REFER_DIR="${DATASET_ROOT}/refer"
  PYTHON_BIN="${PYTHON_BIN:-python3}"

  HF_REPO_ID="${HF_REPO_ID:-MatchLab/ScenePoint}"
  HF_REPO_TYPE="${HF_REPO_TYPE:-dataset}"
  REPO_TYPE="${REPO_TYPE:-dataset}"
  FILENAME_FMT="${FILENAME_FMT:-light_scannet/%s.safetensors}"
  PM_KEY="${PM_KEY:-point_map}"
  RGB_KEY="${RGB_KEY:-color_images}"
  DEVICE="${DEVICE:-cuda}"
  MODEL_ROOT="${MODEL_ROOT:-${PROJECT_ROOT}/src/fg-clip}"

  UNISCENE3D_CKPT="${UNISCENE3D_CKPT:-${PROJECT_ROOT}/results/uniscene3d-base-patch16-224.pth}"
  POMA3D_CKPT="${POMA3D_CKPT:-${PROJECT_ROOT}/results/full_ckpt_100.pth}"
  DFN_CKPT="${DFN_CKPT:-}"
  SIGLIP_CKPT="${SIGLIP_CKPT:-}"
  DFN_MODEL_NAME="${DFN_MODEL_NAME:-hf-hub:apple/DFN2B-CLIP-ViT-B-16}"
  SIGLIP_MODEL_NAME="${SIGLIP_MODEL_NAME:-google/siglip2-base-patch16-224}"
}

run_spatial_bench_module() {
  local module="$1"
  shift
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" -m "${module}" "$@")
}
