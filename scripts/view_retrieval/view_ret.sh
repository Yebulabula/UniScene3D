#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../spatial_bench_common.sh"
setup_spatial_bench_env "${SCRIPT_DIR}"

SCANNET_SCAN_ROOT="${SCANNET_SCAN_ROOT:-${SCAN_ROOT:-}}"
SCANNET_FILENAME_FMT="${SCANNET_FILENAME_FMT:-light_scannet/%s.safetensors}"
BATCH_VIEWS="${BATCH_VIEWS:-32}"

MODULE="evaluator.view_retrieval.zero_shot_eval_view_retrieval"

JSONLS=(
  "${RETRIEVAL_DIR}/scanrefer_retrieval.jsonl"
  "${RETRIEVAL_DIR}/nr3d_retrieval.jsonl"
  "${RETRIEVAL_DIR}/sr3d_retrieval.jsonl"
)

run_eval() {
  local model_type="$1"
  local input_mode="$2"
  local ckpt="$3"
  local jsonl="$4"
  local dataset_name
  local hf_repo_id="${HF_REPO_ID}"

  dataset_name="$(basename "${jsonl}")"

  echo
  echo "=== model=${model_type} input=${input_mode} dataset=${dataset_name} scan_root=${SCANNET_SCAN_ROOT} hf_repo_id=${hf_repo_id:-<disabled>} ==="

  local args=(
    --jsonl "${jsonl}"
    --scan_root "${SCANNET_SCAN_ROOT}"
    --hf_repo_id "${hf_repo_id}"
    --filename_fmt "${SCANNET_FILENAME_FMT}"
    --repo_type "${REPO_TYPE}"
    --pm_key "${PM_KEY}"
    --rgb_key "${RGB_KEY}"
    --device "${DEVICE}"
    --batch_views "${BATCH_VIEWS}"
    --model_type "${model_type}"
    --input_mode "${input_mode}"
    --model_root "${MODEL_ROOT}"
  )

  if [[ -n "${ckpt}" ]]; then
    args+=(--ckpt "${ckpt}")
  fi

  case "${model_type}" in
    dfn)
      args+=(--dfn_model_name "${DFN_MODEL_NAME}")
      ;;
    siglip)
      args+=(--siglip_model_name "${SIGLIP_MODEL_NAME}")
      ;;
  esac

  run_spatial_bench_module "${MODULE}" "${args[@]}"
}

for jsonl in "${JSONLS[@]}"; do
  run_eval uniscene3d pm+image "${UNISCENE3D_CKPT}" "${jsonl}"
  # remove comments to run other models 
  # run_eval fgclip image "" "${jsonl}"
  # run_eval poma3d pm "${POMA3D_CKPT}" "${jsonl}"
  # run_eval dfn image "${DFN_CKPT}" "${jsonl}"
  # run_eval siglip image "${SIGLIP_CKPT}" "${jsonl}"
done
