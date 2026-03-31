#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/../spatial_bench_common.sh"
setup_spatial_bench_env "${SCRIPT_DIR}"

MAX_VIEWS="${MAX_VIEWS:-32}"
BATCH_SCENES="${BATCH_SCENES:-16}"
ROOM_TYPE_JSON="${ROOM_TYPE_JSON:-${CLASSIFICATION_DIR}/scannet_room_types.json}"
NORMALIZE_LABELS="${NORMALIZE_LABELS:---normalize_labels}"
PRINT_CONFUSION_TOPK="${PRINT_CONFUSION_TOPK:-10}"

MODULE="evaluator.scene_classification.zero_shot_scene_cls"

TEMPLATES=(
  "The room type is {room_type}."
  "The scene is a {room_type}."
  "This indoor scene is a {room_type}."
)

run_eval() {
  local model_type="$1"
  local input_mode="$2"
  local ckpt="$3"
  local template="$4"

  echo
  echo "=== model=${model_type} input=${input_mode} template=${template} ==="

  local args=(
    --room_type_json "${ROOM_TYPE_JSON}"
    --hf_repo_id "${HF_REPO_ID}"
    --hf_repo_type "${HF_REPO_TYPE}"
    --filename_fmt "${FILENAME_FMT}"
    --pm_key "${PM_KEY}"
    --rgb_key "${RGB_KEY}"
    --max_views "${MAX_VIEWS}"
    --batch_scenes "${BATCH_SCENES}"
    --templates "${template}"
    --model_type "${model_type}"
    --input_mode "${input_mode}"
    --model_root "${MODEL_ROOT}"
    --dfn_model_name "${DFN_MODEL_NAME}"
    --siglip_model_name "${SIGLIP_MODEL_NAME}"
    --device "${DEVICE}"
    --print_confusion_topk "${PRINT_CONFUSION_TOPK}"
  )

  if [[ -n "${NORMALIZE_LABELS}" ]]; then
    args+=("${NORMALIZE_LABELS}")
  fi

  if [[ -n "${ckpt}" ]]; then
    args+=(--ckpt "${ckpt}")
  fi

  run_spatial_bench_module "${MODULE}" "${args[@]}"
}

for template in "${TEMPLATES[@]}"; do
  # run_eval uniscene3d pm+image "${UNISCENE3D_CKPT}" "${template}"
  run_eval fgclip image "" "${template}"
  run_eval poma3d pm "${POMA3D_CKPT}" "${template}"
  run_eval dfn image "${DFN_CKPT}" "${template}"
  run_eval siglip image "${SIGLIP_CKPT}" "${template}"
done
