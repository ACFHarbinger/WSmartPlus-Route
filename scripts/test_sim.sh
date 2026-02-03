#!/bin/bash

# Default to verbose mode
VERBOSE=true
set -x

# Load Task Config first to get general settings and PROBLEM definition
TASK_CONFIG="assets/configs/tasks/test_sim.yaml"
DATA_CONFIG="assets/configs/data/test_sim.yaml"
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" | grep -v "declare -A")

# Now load the specific environment config based on the problem defined in the task
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" "$ENV_CONFIG" | grep -v "declare -A")
    fi
fi

# MAP ENVIRONMENT VARIABLES TO SCRIPT VARIABLES
if [ -n "$SIM_SIZE" ]; then N_BINS="$SIM_SIZE"; elif [ -n "$SIZE" ]; then N_BINS="$SIZE"; fi
if [ -n "$SIM_WASTE_TYPE" ]; then WTYPE="$SIM_WASTE_TYPE"; elif [ -n "$WTYPE" ]; then WTYPE="$WTYPE"; fi
if [ -n "$SIM_AREA" ]; then AREA="$SIM_AREA"; elif [ -n "$AREA" ]; then AREA="$AREA"; fi
if [ -n "$SIM_PROBLEM" ]; then PROBLEM="$SIM_PROBLEM"; elif [ -n "$PROBLEM" ]; then PROBLEM="$PROBLEM"; fi
if [ -n "$SIM_DAYS" ]; then N_DAYS="$SIM_DAYS"; elif [ -n "$DAYS" ]; then N_DAYS="$DAYS"; fi
if [ -n "$SIM_N_SAMPLES" ]; then N_SAMPLES="$SIM_N_SAMPLES"; elif [ -n "$N_SAMPLES" ]; then N_SAMPLES="$N_SAMPLES"; fi
if [ -n "$SIM_DATA_DISTRIBUTION" ]; then DATA_DIST="$SIM_DATA_DISTRIBUTION"; elif [ -n "$DATA_DISTRIBUTION" ]; then DATA_DIST="$DATA_DISTRIBUTION"; fi
if [ -n "$SIM_POLICIES" ]; then POLICIES=("${SIM_POLICIES[@]}"); fi
if [ -n "$SIM_GATE_PROB_THRESHOLD" ]; then GATE_PROB_THRESHOLD="$SIM_GATE_PROB_THRESHOLD"; fi
if [ -n "$SIM_MASK_PROB_THRESHOLD" ]; then MASK_PROB_THRESHOLD="$SIM_MASK_PROB_THRESHOLD"; fi
if [ -n "$SIM_CHECKPOINT_DAYS" ]; then CHECKPOINTS="$SIM_CHECKPOINT_DAYS"; fi
if [ -n "$EDGE_T" ]; then EDGE_THRESH="$EDGE_T"; fi
if [ -n "$EDGE_M" ]; then EDGE_METHOD="$EDGE_M"; fi
if [ -n "$DIST_M" ]; then DIST_METHOD="$DIST_M"; fi
if [ -n "$VERTEX_M" ]; then VERTEX_METHOD="$VERTEX_M"; fi

while getopts "nc:P:d:N:W:I:M:V:FvC:" flag
do
    case "${flag}" in
        nc) n_cores=${OPTARG};;
        P) AREA=${OPTARG};;
        d) DATA_DIST=${OPTARG};;
        N) N_BINS=${OPTARG};;
        W) WTYPE=${OPTARG};;
        I) POLICIES=(${OPTARG});;
        M) CUSTOM_MODEL_PATH=${OPTARG};;
        V) VEHICLES=${OPTARG};;
        F) RUN_TSP=0;;
        v) VERBOSE=true;;
        C) CONFIG_PATH=${OPTARG};;
    esac
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


# Set n_cores defaults if not set by getopts
if [[ -z $n_cores ]]; then
    n_cores=$N_CORES
fi

# DYNAMIC MODEL PATHS CONSTRUCTION
# Depends on variables (AREA, PROBLEM, etc) which may be from YAML or CLI args
#declare -A MODEL_PATHS
#MODEL_PATHS["amgat"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amgat0"
#MODEL_PATHS["amgat_hrl"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amgat_hrl"
#MODEL_PATHS["amggac"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amggac"
#MODEL_PATHS["amtgc"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amtgc_hrl"

#MODEL_PATH_ARGS=()
#for key in "${!MODEL_PATHS[@]}"; do
#    if [[ -n "$CUSTOM_MODEL_PATH" && "${POLICIES[0]}" == "$key" ]]; then
#        MODEL_PATH_ARGS+=("$key=$CUSTOM_MODEL_PATH")
#    else
#        MODEL_PATH_ARGS+=("$key=${MODEL_PATHS[$key]}")
#    fi
#done

# CONFIG_PATHS is loaded from YAML as associative array

CONFIG_PATH_ARGS=()
for key in "${!CONFIG_PATHS[@]}"; do
    CONFIG_PATH_ARGS+=("$key=${CONFIG_PATHS[$key]}")
done

if [[ -n "$CONFIG_PATH" ]]; then
    # Add manual config path
    CONFIG_PATH_ARGS+=("$CONFIG_PATH")
fi

# Ensure defaults for variables not in YAML or if needing specific logic
# (Most are in YAML now)

# Update dependent paths based on parsed arguments
IDX_PATH="graphs_${N_BINS}V_1N_${WTYPE}.json"
DM_PATH="data/wsr_simulator/distance_matrix/gmaps_distmat_${WTYPE}[${AREA}].csv"
WASTE_PATH="daily_waste/${AREA}${N_BINS}_${DATA_DIST}_wsr${N_DAYS}_N${N_SAMPLES}_seed${SEED}.pkl"
CHECKPOINTS=30

# Real time log
REAL_TIME=1

echo -e "${BLUE}Starting test execution with $n_cores cores...${NC}"
echo -e "${CYAN}----------------------------------------------${NC}"
echo -e "${CYAN}[PARAM]${NC} Cores:         ${MAGENTA}$n_cores${NC}"
echo -e "${CYAN}[PARAM]${NC} Problem:       ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[PARAM]${NC} Area:          ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[PARAM]${NC} Policies:      ${MAGENTA}$POLICIES${NC}"
echo -e "${CYAN}[PARAM]${NC} Samples:       ${MAGENTA}$N_SAMPLES${NC}"
echo -e "${CYAN}[PARAM]${NC} Days:          ${MAGENTA}$N_DAYS${NC}"
echo -e "${CYAN}[PARAM]${NC} Real Time Log: ${MAGENTA}$REAL_TIME${NC}"
echo -e "${CYAN}----------------------------------------------${NC}"
echo ""
echo ""


# Helper to join array with commas
join_by() {
  local d=${1-} f=${2-}
  if shift 2; then
    printf %s "$f" "${@/#/$d}"
  fi
}

quote_list() {
    local -n arr=$1
    local quoted=()
    for item in "${arr[@]}"; do
        quoted+=("'$item'")
    done
    join_by , "${quoted[@]}"
}

POLICIES_STR="[$(quote_list POLICIES)]"


# Build command arguments array
# Helper to add arg if value exists
add_arg() {
    local key=$1
    local value=$2
    if [[ -n "$value" ]]; then
        CMD_ARGS+=("$key=$value")
    fi
}

CMD_ARGS=()
add_arg "sim.policies" "${POLICIES_STR}"
add_arg "sim.data_distribution" "$DATA_DIST"
add_arg "sim.cpu_cores" "$n_cores"
add_arg "sim.n_samples" "$N_SAMPLES"
add_arg "sim.area" "$AREA"
add_arg "sim.bin_idx_file" "$IDX_PATH"
add_arg "sim.size" "$N_BINS"
add_arg "sim.seed" "$SEED"
add_arg "sim.problem" "$PROBLEM"
add_arg "sim.n_vehicles" "$VEHICLES"
add_arg "sim.vertex_method" "$VERTEX_METHOD"
add_arg "sim.distance_method" "$DIST_METHOD"
add_arg "sim.edge_threshold" "$EDGE_THRESH"
add_arg "sim.edge_method" "$EDGE_METHOD"
add_arg "sim.waste_type" "$WTYPE"
add_arg "sim.env_file" "$ENV_FILE"
add_arg "sim.gplic_file" "$GP_LIC_FILE"
add_arg "sim.gapik_file" "$GOOGLE_API_FILE"
add_arg "sim.waste_filepath" "$WASTE_PATH"
add_arg "sim.symkey_name" "$SYM_KEY"
if [[ -n "$DM_PATH" ]]; then
    CMD_ARGS+=("sim.dm_filepath='$DM_PATH'")
fi
add_arg "sim.days" "$N_DAYS"
add_arg "sim.checkpoint_days" "$CHECKPOINTS"
add_arg "sim.gate_prob_threshold" "$GATE_PROB_THRESHOLD"
add_arg "sim.mask_prob_threshold" "$MASK_PROB_THRESHOLD"
add_arg "sim.log_level" "${LOG_LEVEL:-WARNING}"

# Conditionally add optional arguments
if [ -n "$STATS_PATH" ]; then
    CMD_ARGS+=("sim.stats_filepath=$STATS_PATH")
fi

if [ ${#MODEL_PATH_ARGS[@]} -gt 0 ]; then
    CMD_ARGS+=("sim.model_path=${MODEL_PATH_ARGS[@]}")
fi

if [ "$REAL_TIME" -eq 0 ]; then
    echo -e "${BLUE}Running with real_time_log...${NC}"
    CMD_ARGS+=("sim.real_time_log=true")
    add_arg "sim.decode_type" "$DECODE_TYPE"
else
    echo -e "${BLUE}Running without real_time_log...${NC}"
    add_arg "sim.decode_type" "$DECODE_TYPE"
fi

# Execute
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4
    exec 3>&- 4>&-
fi

uv run python main.py test_sim "${CMD_ARGS[@]}"

if [ "$VERBOSE" = false ]; then
    exec >/dev/null 2>&1
fi

echo ""
echo -e "${GREEN}✓ [SUCCESS] Simulation test completed successfully.${NC}"
