#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../spatial_bench_common.sh"
setup_spatial_bench_env "${SCRIPT_DIR}"

SCANNET_SCAN_ROOT="${SCANNET_SCAN_ROOT:-${SCAN_ROOT:-}}"
SCANNETPP_SCAN_ROOT="${SCANNETPP_SCAN_ROOT:-}"
SCANNET_VAL_SPLIT="${SCANNET_VAL_SPLIT:-${PROJECT_ROOT}/dataset/ScanNet/annotations/splits/scannetv2_val.txt}"
SCANNETPP_VAL_SPLIT="${SCANNETPP_VAL_SPLIT:-${SCANNETPP_SCAN_ROOT:+${SCANNETPP_SCAN_ROOT%/}/../splits/nvs_sem_val.txt}}"
SCANNET_FILENAME_FMT="${SCANNET_FILENAME_FMT:-light_scannet/%s.safetensors}"
SCANNETPP_FILENAME_FMT="${SCANNETPP_FILENAME_FMT:-light_scannetpp/%s.safetensors}"
MAX_VIEWS="${MAX_VIEWS:-32}"
N_UTTERANCES="${N_UTTERANCES:-10}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-2}"
STRATEGY="${STRATEGY:-first}"
MODULE="evaluator.scene_retrieval.zero_shot_eval_scene_retrieval"

JSONLS=(
  "${RETRIEVAL_DIR}/scannetpp_retrieval.jsonl"
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
  local scan_root
  local filename_fmt
  local split_file
  local hf_repo_id="${HF_REPO_ID}"

  dataset_name="$(basename "${jsonl}")"
  case "${dataset_name}" in
    scannetpp_retrieval.jsonl)
      scan_root="${SCANNETPP_SCAN_ROOT}"
      filename_fmt="${SCANNETPP_FILENAME_FMT}"
      split_file="${SCANNETPP_VAL_SPLIT}"
      ;;
    *)
      scan_root="${SCANNET_SCAN_ROOT}"
      filename_fmt="${SCANNET_FILENAME_FMT}"
      split_file="${SCANNET_VAL_SPLIT}"
      ;;
  esac

  echo
  echo "=== model=${model_type} input=${input_mode} dataset=${dataset_name} scan_root=${scan_root} hf_repo_id=${hf_repo_id:-<disabled>} ==="

  local args=(
    --jsonl "${jsonl}"
    --scan_root "${scan_root}"
    --hf_repo_id "${hf_repo_id}"
    --hf_repo_type "${HF_REPO_TYPE}"
    --filename_fmt "${filename_fmt}"
    --pm_key "${PM_KEY}"
    --max_views "${MAX_VIEWS}"
    --n_utterances "${N_UTTERANCES}"
    --batch_size "${BATCH_SIZE}"
    --num_workers "${NUM_WORKERS}"
    --strategy "${STRATEGY}"
    --model_type "${model_type}"
    --input_mode "${input_mode}"
    --model_root "${MODEL_ROOT}"
    --dfn_model_name "${DFN_MODEL_NAME}"
    --siglip_model_name "${SIGLIP_MODEL_NAME}"
    --device "${DEVICE}"
  )

  if [[ -n "${split_file}" ]]; then
    args+=(--split_file "${split_file}")
  fi

  if [[ -n "${ckpt}" ]]; then
    args+=(--ckpt "${ckpt}")
  fi

  run_spatial_bench_module "${MODULE}" "${args[@]}"
}

for jsonl in "${JSONLS[@]}"; do
  run_eval uniscene3d pm+image "${UNISCENE3D_CKPT}" "${jsonl}"
  run_eval fgclip image "" "${jsonl}"
  run_eval poma3d pm "${POMA3D_CKPT}" "${jsonl}"
  run_eval dfn image "${DFN_CKPT}" "${jsonl}"
  run_eval siglip image "${SIGLIP_CKPT}" "${jsonl}"
done
