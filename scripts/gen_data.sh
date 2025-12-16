#!/bin/bash

# Default to quiet mode
VERBOSE=false

# Handle --verbose if it appears after other arguments
for arg in "$@"; do
    if [[ "$arg" == "--verbose" ]]; then
        VERBOSE=true
    fi
done

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2  # Save original stdout (fd1) to fd3, stderr (fd2) to fd4
    exec >/dev/null 2>&1
fi

SEED=42
START=0
N_EPOCHS=31
PROBLEM="vrpp"
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
N_VAL_DATA=1280
N_TEST_DATA=10
DATASET_NAME="time"
VAL_DATASET_NAME="${DATASET_NAME}_val"
TEST_DATASET_NAME="wsr"
DATA_DISTS=("gamma1")

GENERATE_DATASET=0
GENERATE_VAL_DATASET=1
GENERATE_TEST_DATASET=1

echo "Starting data generation..."

if [ "$GENERATE_DATASET" -eq 0 ]; then
    echo "Generating main dataset..."
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
    echo "Generating validation dataset..."
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
    echo "Generating test dataset..."
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

echo "Done!"