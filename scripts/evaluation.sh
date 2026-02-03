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

# ==============================================================================

# ==============================================================================
# DEFAULT CONFIGURATION
# Load default values from YAML
# ==============================================================================

# Load Task Config first to get general settings and PROBLEM definition
CONFIG_FILE="assets/configs/tasks/evaluation.yaml"
DATA_CONFIG="assets/configs/data/eval_data.yaml"
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$CONFIG_FILE" "$DATA_CONFIG")

# Now load the specific environment config based on the problem defined in the task
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$CONFIG_FILE" "$DATA_CONFIG" "$ENV_CONFIG")
    fi
fi

# MAP ENVIRONMENT VARIABLES TO SCRIPT VARIABLES
# The environment config (e.g. cwcvrp.yaml) exports variables like SIZE, WTYPE, etc.
# The evaluation script expects GRAPH_SIZE, WASTE_TYPE, etc.
if [ -n "$SIZE" ]; then GRAPH_SIZE="$SIZE"; fi
if [ -n "$AREA" ]; then AREA="$AREA"; fi # Redundant but safe
if [ -n "$WTYPE" ]; then WASTE_TYPE="$WTYPE"; fi
if [ -n "$EDGE_T" ]; then EDGE_THRESHOLD="$EDGE_T"; fi
if [ -n "$EDGE_M" ]; then EDGE_METHOD="$EDGE_M"; fi
if [ -n "$DIST_M" ]; then DISTANCE_METHOD="$DIST_M"; fi
if [ -n "$VERTEX_M" ]; then VERTEX_METHOD="$VERTEX_M"; fi


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
    uv run python main.py eval
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
    --max_calc_batch_size "$MAX_CALC_BATCH_SIZE"
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

echo -e "${BLUE}╔══════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       ALGORITHM EVALUATION MODULE        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════╝${NC}"
echo -e "${CYAN}[PARAM]${NC} Model:            ${MAGENTA}$MODEL_PATH${NC}"
echo -e "${CYAN}[PARAM]${NC} Datasets:         ${MAGENTA}${DATASET_ARGS}${NC}"
echo -e "${CYAN}[PARAM]${NC} Decode Strategy:  ${MAGENTA}$DECODE_STRATEGY${NC}"
echo -e "${CYAN}[PARAM]${NC} Width:            ${MAGENTA}${WIDTH_ARGS}${NC}"
echo -e "${CYAN}[PARAM]${NC} Output File:      ${MAGENTA}${OUTPUT_FILE:-None}${NC}"
echo ""
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
