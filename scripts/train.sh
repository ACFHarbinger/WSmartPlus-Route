#!/bin/bash

# ==============================================================================
# TRAINING SCRIPT (Config-File-Only Approach)
# ==============================================================================
# This script invokes main.py train_lightning with config values from YAML.
# All configuration is defined in:
#   - assets/configs/tasks/train.yaml
#   - assets/configs/data/train_data.yaml
#   - assets/configs/envs/${PROBLEM}.yaml
#   - assets/configs/models/${MODEL}.yaml
# ==============================================================================

set -e

# Handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default to verbose mode
VERBOSE=true

# Configuration files
TASK_CONFIG="assets/configs/tasks/train.yaml"
DATA_CONFIG="assets/configs/data/train_data.yaml"

# Load Task Config first
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true

# Load environment config based on problem
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" "$ENV_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true
    fi
fi

# Parse CLI overrides
CLI_OVERRIDES=()
while getopts "qm:e:s:n:" flag; do
    case "${flag}" in
        q) VERBOSE=false;;
        m) MODEL_NAME="${OPTARG}";;
        e) EPOCHS="${OPTARG}";;
        s) SIZE="${OPTARG}";;
        n) N_DATA="${OPTARG}";;
        \?) echo -e "${RED}Invalid option: -${OPTARG}${NC}" >&2; exit 1;;
    esac
done
shift $((OPTIND-1))

# Use loaded or default values
PROBLEM="${ENV_NAME:-${PROBLEM:-cwcvrp}}"
AREA="${AREA:-riomaior}"
SIZE="${ENV_NUM_LOC:-${SIZE:-50}}"
EPOCHS="${TRAIN_N_EPOCHS:-${EPOCHS:-100}}"
B_SIZE="${TRAIN_BATCH_SIZE:-128}"
SEED="${SEED:-42}"
MODEL="${MODEL_NAME:-am}"
ENCODER="${MODEL_ENCODER_TYPE:-gat}"

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       TRAINING MODULE (Hydra-based)      ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}${PROBLEM}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph Size: ${MAGENTA}${SIZE}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}${AREA}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Epochs:     ${MAGENTA}${EPOCHS}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Model:      ${MAGENTA}${MODEL}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Encoder:    ${MAGENTA}${ENCODER}${NC}"
echo ""

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2
    exec >/dev/null 2>&1
fi

# Execute with config values from YAML
uv run python main.py train_lightning \
    "env.name='${PROBLEM}'" \
    "env.num_loc=${SIZE}" \
    "env.area='${AREA}'" \
    "model.name='${MODEL}'" \
    "model.encoder_type='${ENCODER}'" \
    "train.n_epochs=${EPOCHS}" \
    "train.batch_size=${B_SIZE}" \
    "seed=${SEED}" \
    "${CLI_OVERRIDES[@]}" \
    "$@"

# Restore output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

echo ""
echo -e "${GREEN}✓ [DONE] Training completed.${NC}"
