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

# ==============================================================================
# DEFAULT CONFIGURATION
# Set default values for all evaluation parameters
# ==============================================================================

# Core Evaluation Settings
DECODE_TYPE="greedy"
DECODE_STRATEGY="greedy"  # Defaulting to greedy for simplicity
SOFTMAX_TEMPERATURE=1.0
VAL_SIZE=12800
OFFSET=0
EVAL_BATCH_SIZE=256
MAX_CALC_BATCH_SIZE=12800
RESULTS_DIR="results_eval"
MODEL_PATH="checkpoints/best_model.pt"

# Data/Dataset Configuration
DATASETS=("data/example_vrp_50.pkl") # Array for multiple datasets
WIDTH=(0) # Array for multiple beam sizes (0 to disable beam search)
GRAPH_SIZE=50
AREA="riomaior"
WASTE_TYPE="plastic"

# Graph Configuration
FOCUS_GRAPH=
FOCUS_SIZE=0
EDGE_THRESHOLD='0'
EDGE_METHOD="knn" # Assumed a common default
DISTANCE_METHOD="ogd"
VERTEX_METHOD="mmn"

# Flags (Default to false/unset)
OVERWRITE_FLAG="" # -f
NO_CUDA_FLAG=""
NO_PROGRESS_BAR_FLAG=""
COMPRESS_MASK_FLAG=""
MULTIPROCESSING_FLAG=""

# Optional Output
OUTPUT_FILE=""

# ==============================================================================
# COMMAND LINE ARGUMENT PARSING
# Allows overriding defaults using flags
# ==============================================================================

# Parse options: m=model, D=datasets, O=output, f=overwrite, c=no_cuda, w=width
while getopts "m:D:O:fw:nC:p" flag
do
    case "${flag}" in
        m) MODEL_PATH=${OPTARG};;
        D) DATASETS=(${OPTARG});;
        O) OUTPUT_FILE=${OPTARG};;
        f) OVERWRITE_FLAG="-f";;
        w) WIDTH=(${OPTARG});;
        C) DECODE_STRATEGY=${OPTARG};;
        n) NO_CUDA_FLAG="--no_cuda";;
        p) NO_PROGRESS_BAR_FLAG="--no_progress_bar";;
        \?) echo "Invalid option: -${OPTARG}" >&2; exit 1;;
    esac
done

# Shift off the options (if any) to allow positional arguments later, though none are expected
shift $((OPTIND-1))


# ==============================================================================
# CONSTRUCT COMMAND
# Build the final Python evaluation command
# ==============================================================================

# Ensure WIDTH is joined into a space-separated string for the command
WIDTH_ARGS="${WIDTH[@]}"
DATASET_ARGS="${DATASETS[@]}"

# Conditional arguments
if [[ -n "$OUTPUT_FILE" ]]; then
    OUTPUT_ARG="-o $OUTPUT_FILE"
fi

# Multi-word string argument construction (optional flags are added as separate arguments)
PYTHON_CMD=(
    python main.py eval
    --model "$MODEL_PATH"
    --datasets ${DATASET_ARGS}
    $OVERWRITE_FLAG
    $OUTPUT_ARG
    --val_size "$VAL_SIZE"
    --offset "$OFFSET"
    --eval_batch_size "$EVAL_BATCH_SIZE"
    --decode_type "$DECODE_TYPE"
    --width ${WIDTH_ARGS}
    --decode_strategy "$DECODE_STRATEGY"
    --softmax_temperature "$SOFTMAX_TEMPERATURE"
    $NO_CUDA_FLAG
    $NO_PROGRESS_BAR_FLAG
    $COMPRESS_MASK_FLAG # Disabled by default, set as a flag manually if needed
    --max_calc_batch_size "$MAX_CALC_BATCH_size"
    --results_dir "$RESULTS_DIR"
    $MULTIPROCESSING_FLAG # Disabled by default, set as a flag manually if needed
    --graph_size "$GRAPH_SIZE"
    --area "$AREA"
    --waste_type "$WASTE_TYPE"
    --focus_size "$FOCUS_SIZE"
    --edge_threshold "$EDGE_THRESHOLD"
    --edge_method "$EDGE_METHOD"
    --distance_method "$DISTANCE_METHOD"
    --vertex_method "$VERTEX_METHOD"
)

# Add focus_graph only if it is explicitly set
if [[ -n "$FOCUS_GRAPH" ]]; then
    PYTHON_CMD+=(--focus_graph "$FOCUS_GRAPH")
fi


# ==============================================================================
# EXECUTION
# ==============================================================================

echo "=========================================="
echo "Starting Algorithm Evaluation"
echo "Model: $MODEL_PATH"
echo "Datasets: ${DATASET_ARGS}"
echo "Decode Strategy: $DECODE_STRATEGY (Width: ${WIDTH_ARGS})"
echo "Output File: ${OUTPUT_FILE:-None}"
echo "=========================================="
echo ""

# Execute the command
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
    exec 3>&- 4>&-  # Close the temporary file descriptors
fi
"${PYTHON_CMD[@]}"
if [ "$VERBOSE" = false ]; then
    exec >/dev/null 2>&1
fi

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Evaluation completed successfully."
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "ERROR: Evaluation failed." >&2
    echo "=========================================="
    exit 1
fi