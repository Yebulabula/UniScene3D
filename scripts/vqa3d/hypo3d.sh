#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# ==== USER SETTINGS ====
CONFIG="configs/finetune/hypo3d_finetune.yaml"
NOTE="hypo3d_sft_align_run1"
EXP_NAME="hypo3d_sft_align_run1"
cd "${PROJECT_ROOT}"

# ==== SAFETY ====
set -e
set -o pipefail

# ==== OUTPUT LOGGING ====
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR="logs"
mkdir -p "$LOGDIR"
LOGFILE="$LOGDIR/${EXP_NAME}_${NOTE}_$TIMESTAMP.log"
# ==== EXPERIMENT DIRECTORY ====
EXP_DIR="results/${EXP_NAME}_${NOTE}"

echo "[INFO] Starting training: $EXP_NAME ($NOTE)"
echo "[INFO] Logging to: $LOGFILE"
echo "[INFO] Experiment directory: $EXP_DIR"

export TOKENIZERS_PARALLELISM=false

# ==== LAUNCH ====
python launch.py --mode accelerate --gpu_per_node 1 --num_nodes 1 \
    --config "$CONFIG" \
    note="$NOTE" \
    name="$EXP_NAME"
