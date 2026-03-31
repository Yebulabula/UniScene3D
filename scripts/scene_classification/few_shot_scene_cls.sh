#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../spatial_bench_common.sh"
setup_spatial_bench_env "${SCRIPT_DIR}"

MAX_VIEWS="${MAX_VIEWS:-32}"
ROOM_TYPE_JSON="${ROOM_TYPE_JSON:-${CLASSIFICATION_DIR}/scannet_room_types.json}"
NORMALIZE_LABELS="${NORMALIZE_LABELS:---normalize_labels}"

N_TRAIN_PER_CLASS="${N_TRAIN_PER_CLASS:-10}"
N_VAL_PER_CLASS="${N_VAL_PER_CLASS:-10}"
SEED="${SEED:-0}"
LAMBDA_MIN="${LAMBDA_MIN:-1e-6}"
LAMBDA_MAX="${LAMBDA_MAX:-1e6}"
LAMBDA_STEPS="${LAMBDA_STEPS:-96}"
MAX_ITER="${MAX_ITER:-1000}"

MODULE="evaluator.scene_classification.few_shot_scene_cls"

run_eval() {
  local model_type="$1"
  local input_mode="$2"
  local ckpt="$3"

  echo
  echo "=== few-shot model=${model_type} input=${input_mode} ==="

  local args=(
    --room_type_json "${ROOM_TYPE_JSON}"
    --hf_repo_id "${HF_REPO_ID}"
    --hf_repo_type "${HF_REPO_TYPE}"
    --filename_fmt "${FILENAME_FMT}"
    --pm_key "${PM_KEY}"
    --rgb_key "${RGB_KEY}"
    --max_views "${MAX_VIEWS}"
    --model_type "${model_type}"
    --input_mode "${input_mode}"
    --model_root "${MODEL_ROOT}"
    --dfn_model_name "${DFN_MODEL_NAME}"
    --siglip_model_name "${SIGLIP_MODEL_NAME}"
    --device "${DEVICE}"
    --n_train_per_class "${N_TRAIN_PER_CLASS}"
    --n_val_per_class "${N_VAL_PER_CLASS}"
    --seed "${SEED}"
    --lambda_min "${LAMBDA_MIN}"
    --lambda_max "${LAMBDA_MAX}"
    --lambda_steps "${LAMBDA_STEPS}"
    --max_iter "${MAX_ITER}"
  )

  if [[ -n "${NORMALIZE_LABELS}" ]]; then
    args+=("${NORMALIZE_LABELS}")
  fi

  if [[ -n "${ckpt}" ]]; then
    args+=(--ckpt "${ckpt}")
  fi

  run_spatial_bench_module "${MODULE}" "${args[@]}"
}

# run_eval uniscene3d pm+image "${UNISCENE3D_CKPT}"
run_eval fgclip image ""
run_eval poma3d pm "${POMA3D_CKPT}"
run_eval dfn image "${DFN_CKPT}"
run_eval siglip image "${SIGLIP_CKPT}"
