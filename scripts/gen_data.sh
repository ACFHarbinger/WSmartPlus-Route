#!/bin/bash

# Default to verbose mode
VERBOSE=true

# Handle --quiet if it appears after other arguments
for arg in "$@"; do
    if [[ "$arg" == "--quiet" ]]; then
        VERBOSE=false
    fi
done


# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2  # Save original stdout (fd1) to fd3, stderr (fd2) to fd4
    exec >/dev/null 2>&1
fi

# Load Task Config first to get general settings and PROBLEM definition
TASK_CONFIG="assets/configs/tasks/gen_data.yaml"
DATA_CONFIG="assets/configs/data/gen_data.yaml"
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG")

# Now load the specific environment config based on the problem defined in the task
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" "$ENV_CONFIG")
    fi
fi

# MAP ENVIRONMENT VARIABLES TO SCRIPT VARIABLES
if [ -n "$WTYPE" ]; then WTYPE="$WTYPE"; fi # Already set if loaded env
if [ -n "$AREA" ]; then AREA="$AREA"; fi
if [ -n "$VERTEX_M" ]; then VERTEX_METHOD="$VERTEX_M"; fi

# Derived Variables
FOCUS_GRAPHS=()
for size in "${SIZES[@]}"; do
    FOCUS_GRAPHS+=("data/wsr_simulator/bins_selection/graphs_${size}V_1N_${WTYPE}.json")
done

VAL_DATASET_NAME="${DATASET_NAME}_val"

echo -e "${BLUE}Starting data generation module...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Sizes:      ${MAGENTA}${SIZES[*]}${NC}"
echo -e "${CYAN}[CONFIG]${NC} Waste Type: ${MAGENTA}$WTYPE${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

if [ "$GENERATE_DATASET" -eq 0 ]; then
    echo -e "${BLUE}Generating main dataset...${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    uv run python main.py gen_data --name "$DATASET_NAME" --problem "$PROBLEM" -f \
    --waste_type "$WTYPE" --graph_sizes "${SIZES[@]}" --dataset_size "$N_DATA" \
    --focus_graph "${FOCUS_GRAPHS[@]}" --focus_size "$FOCUS_SIZE" --data_dir "$DATA_DIR" \
    --area "$AREA" --vertex_method "$VERTEX_METHOD" --epoch_start "$START" --seed "$SEED" \
    --n_epochs "$N_EPOCHS" --data_distribution "${DATA_DISTS[@]}" --dataset_type "$D_TYPE";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
fi

if [ "$GENERATE_VAL_DATASET" -eq 0 ]; then
    echo -e "${BLUE}Generating validation dataset...${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    uv run python main.py gen_data --name "$VAL_DATASET_NAME" --problem "$PROBLEM" -f \
    --waste_type "$WTYPE" --graph_sizes "${SIZES[@]}" --dataset_size "$N_VAL_DATA" \
    --area "$AREA" --vertex_method "$VERTEX_METHOD" --epoch_start "$START" --seed "$SEED" \
    --n_epochs "$N_EPOCHS" --data_distribution "${DATA_DISTS[@]}" --dataset_type "$D_TYPE" \
    --focus_graph "${FOCUS_GRAPHS[@]}" --focus_size "$VAL_FOCUS_SIZE" --data_dir "$DATA_DIR";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
fi

if [ "$GENERATE_TEST_DATASET" -eq 0 ]; then
    echo -e "${BLUE}Generating test dataset...${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    uv run python main.py gen_data --name "$TEST_DATASET_NAME" --problem "$PROBLEM" -f \
    --area "$AREA" --vertex_method "$VERTEX_METHOD" --epoch_start "$START" --seed "$SEED" \
    --n_epochs "$N_EPOCHS" --data_distribution "${DATA_DISTS[@]}" --dataset_type "$D_TYPE_SIM" \
    --focus_graph "${FOCUS_GRAPHS[@]}" --focus_size "$TEST_FOCUS_SIZE" --data_dir "$SIM_DATA_DIR" \
    --waste_type "$WTYPE" --graph_sizes "${SIZES[@]}" --dataset_size "$N_TEST_DATA";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
fi

echo -e "${GREEN}✓ [DONE] Data generation process completed.${NC}"
