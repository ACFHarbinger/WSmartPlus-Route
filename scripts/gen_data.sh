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

SEED=42
START=0
N_EPOCHS=150
PROBLEM="wcvrp"
AREA="riomaior"
VERTEX_METHOD="mmn"

DATA_DIR="datasets"
SIM_DATA_DIR="daily_waste"
D_TYPE="train_time" #"train"
D_TYPE_SIM="test_simulator"

WTYPE="plastic"
FOCUS_SIZE=1280
VAL_FOCUS_SIZE=1280
TEST_FOCUS_SIZE=1
SIZES=(20 50 100 170)
for size in "${SIZES[@]}"; do
    FOCUS_GRAPHS+=("graphs_${size}V_1N_${WTYPE}.json")
done

N_DATA=1280
N_VAL_DATA=128
N_TEST_DATA=1
DATASET_NAME="time"
VAL_DATASET_NAME="${DATASET_NAME}_val"
TEST_DATASET_NAME="wsr"
DATA_DISTS=("gamma1")

GENERATE_DATASET=0
GENERATE_VAL_DATASET=0
GENERATE_TEST_DATASET=1

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
    python main.py gen_data --name "$DATASET_NAME" --problem "$PROBLEM" -f \
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
    python main.py gen_data --name "$VAL_DATASET_NAME" --problem "$PROBLEM" -f \
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
    python main.py gen_data --name "$TEST_DATASET_NAME" --problem "$PROBLEM" -f \
    --area "$AREA" --vertex_method "$VERTEX_METHOD" --epoch_start "$START" --seed "$SEED" \
    --n_epochs "$N_EPOCHS" --data_distribution "${DATA_DISTS[@]}" --dataset_type "$D_TYPE_SIM" \
    --focus_graph "${FOCUS_GRAPHS[@]}" --focus_size "$TEST_FOCUS_SIZE" --data_dir "$SIM_DATA_DIR" \
    --waste_type "$WTYPE" --graph_sizes "${SIZES[@]}" --dataset_size "$N_TEST_DATA";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
fi

echo -e "${GREEN}✓ [DONE] Data generation process completed.${NC}"
