#!/bin/bash

# ==============================================================================
# DATA GENERATION SCRIPT (Config-File-Only Approach)
# ==============================================================================
# This script invokes main.py gen_data with config values loaded from YAML.
# Configuration is defined in:
#   - assets/configs/tasks/gen_data.yaml
#   - assets/configs/data/gen_data.yaml
# ==============================================================================

set -e

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
TASK_CONFIG="assets/configs/tasks/gen_data.yaml"
DATA_CONFIG="assets/configs/data/gen_data.yaml"

# Load config
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true

# Load environment config based on problem
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" "$ENV_CONFIG" 2>/dev/null | grep -v "declare -A") 2>/dev/null || true
    fi
fi

# Parse CLI overrides
while getopts "q" flag; do
    case "${flag}" in
        q) VERBOSE=false;;
        \?) echo -e "${RED}Invalid option: -${OPTARG}${NC}" >&2; exit 1;;
    esac
done
shift $((OPTIND-1))

# Use loaded or default values
PROBLEM="${PROBLEM:-cwcvrp}"
AREA="${AREA:-riomaior}"
WTYPE="${WTYPE:-plastic}"
SIZES="${SIZES[@]:-50}"
SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-data/datasets}"
N_DATA="${N_DATA:-1280}"

# Derived values
FOCUS_GRAPHS=()
for size in ${SIZES[@]}; do
    FOCUS_GRAPHS+=("data/wsr_simulator/bins_selection/graphs_${size}V_1N_${WTYPE}.json")
done

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       DATA GENERATION MODULE             ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}${PROBLEM}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}${AREA}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Sizes:      ${MAGENTA}${SIZES[*]}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Waste Type: ${MAGENTA}${WTYPE}${NC}"
echo ""

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2
    exec >/dev/null 2>&1
fi

# Generate main dataset
if [ "${GENERATE_DATASET:-0}" -eq 0 ]; then
    echo -e "${BLUE}Generating main dataset...${NC}"
    uv run python main.py gen_data \
        --name "${DATASET_NAME:-train}" \
        --problem "$PROBLEM" \
        -f \
        --waste_type "$WTYPE" \
        --graph_sizes ${SIZES[@]} \
        --dataset_size "$N_DATA" \
        --focus_graph "${FOCUS_GRAPHS[@]}" \
        --data_dir "$DATA_DIR" \
        --area "$AREA" \
        --seed "$SEED" \
        "$@"
fi

# Generate validation dataset
if [ "${GENERATE_VAL_DATASET:-0}" -eq 0 ]; then
    echo -e "${BLUE}Generating validation dataset...${NC}"
    uv run python main.py gen_data \
        --name "${DATASET_NAME:-train}_val" \
        --problem "$PROBLEM" \
        -f \
        --waste_type "$WTYPE" \
        --graph_sizes ${SIZES[@]} \
        --dataset_size "${N_VAL_DATA:-1280}" \
        --data_dir "$DATA_DIR" \
        --area "$AREA" \
        --seed "$SEED" \
        "$@"
fi

# Generate test dataset
if [ "${GENERATE_TEST_DATASET:-0}" -eq 0 ]; then
    echo -e "${BLUE}Generating test dataset...${NC}"
    uv run python main.py gen_data \
        --name "${TEST_DATASET_NAME:-test}" \
        --problem "$PROBLEM" \
        -f \
        --area "$AREA" \
        --vertex_method "${VERTEX_METHOD:-mmn}" \
        --epoch_start "${START:-0}" \
        --seed "$SEED" \
        --n_epochs "${N_EPOCHS:-1}" \
        --data_distribution "${DATA_DISTS[@]}" \
        --dataset_type "${D_TYPE_SIM:-test_time}" \
        --focus_graph "${FOCUS_GRAPHS[@]}" \
        --focus_size "${TEST_FOCUS_SIZE:-1280}" \
        --data_dir "${SIM_DATA_DIR:-$DATA_DIR}" \
        --waste_type "$WTYPE" \
        --graph_sizes "${SIZES[@]}" \
        --dataset_size "${N_TEST_DATA:-1280}" \
        "$@"
fi

# Restore output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

echo ""
echo -e "${GREEN}✓ [DONE] Data generation process completed.${NC}"
