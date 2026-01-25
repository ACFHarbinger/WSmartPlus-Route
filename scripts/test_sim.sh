#!/bin/bash

# Default to verbose mode
VERBOSE=true

# Load Task Config first to get general settings and PROBLEM definition
TASK_CONFIG="scripts/configs/tasks/test_sim.yaml"
DATA_CONFIG="scripts/configs/data/test_sim.yaml"
eval $(uv run python scripts/utils/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG")

# Now load the specific environment config based on the problem defined in the task
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="scripts/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        eval $(uv run python scripts/utils/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG" "$ENV_CONFIG")
    fi
fi

# MAP ENVIRONMENT VARIABLES TO SCRIPT VARIABLES
if [ -n "$SIZE" ]; then N_BINS="$SIZE"; fi
if [ -n "$WTYPE" ]; then WTYPE="$WTYPE"; fi
if [ -n "$AREA" ]; then AREA="$AREA"; fi
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
declare -A MODEL_PATHS
MODEL_PATHS["amgat"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amgat0"
MODEL_PATHS["amgat_hrl"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amgat_hrl"
MODEL_PATHS["amggac"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amggac"
MODEL_PATHS["amtgc"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amtgc_hrl"

MODEL_PATH_ARGS=()
for key in "${!MODEL_PATHS[@]}"; do
    if [[ -n "$CUSTOM_MODEL_PATH" && "${POLICIES[0]}" == "$key" ]]; then
        MODEL_PATH_ARGS+=("$key=$CUSTOM_MODEL_PATH")
    else
        MODEL_PATH_ARGS+=("$key=${MODEL_PATHS[$key]}")
    fi
done

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
WASTE_PATH=""
CHECKPOINTS=30

# Route postprocessing
RUN_TSP=1
TWO_OPT_MAX_ITER=0

echo -e "${BLUE}Starting test execution with $n_cores cores...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[PARAM]${NC} Cores:      ${MAGENTA}$n_cores${NC}"
echo -e "${CYAN}[PARAM]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[PARAM]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[PARAM]${NC} Policies:   ${MAGENTA}$POLICIES${NC}"
echo -e "${CYAN}[PARAM]${NC} Samples:    ${MAGENTA}$N_SAMPLES${NC}"
echo -e "${CYAN}[PARAM]${NC} Days:       ${MAGENTA}$N_DAYS${NC}"
echo -e "${CYAN}[PARAM]${NC} Fast TSP:   ${MAGENTA}$RUN_TSP${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""
echo ""

# Add option --real_time_log to the command line if you want real time updates of the simulation
if [ "$RUN_TSP" -eq 0 ]; then
    echo -e "${BLUE}Running with fast_tsp...${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    uv run python main.py test_sim --policies "${POLICIES[@]}" --data_distribution "$DATA_DIST" --run_tsp \
    --n_samples "$N_SAMPLES" --bin_idx_file "$IDX_PATH" --size "$N_BINS" --seed "$SEED" --area "$AREA" \
    --n_vehicles "$VEHICLES" --vm "$VERTEX_METHOD" --cpd "$CHECKPOINTS" --gplic_file "$GP_LIC_FILE" \
    --lvl "${REGULAR_LEVEL[@]}" --cf "${LAST_MINUTE_CF[@]}" --gp "${GUROBI_PARAM[@]}" --dt "$DECODE_TYPE" \
    --lac "${LOOKAHEAD_CONFIGS[@]}" --hp "${HEXALY_PARAM[@]}" --problem "$PROBLEM" --days "$N_DAYS" \
    --waste_type "$WTYPE" --cc "$n_cores" --et "$EDGE_THRESH" --em "$EDGE_METHOD" --env_file "$ENV_FILE" \
    --gapik_file "$GOOGLE_API_FILE" --symkey_name "$SYM_KEY" --dm_filepath "$DM_PATH" --dm "$DIST_METHOD" \
    --waste_filepath "$WASTE_PATH" --stats_filepath "$STATS_PATH" --model_path "${MODEL_PATH_ARGS[@]}" \
    --gate_prob_threshold $GATE_PROB_THRESHOLD --mask_prob_threshold $MASK_PROB_THRESHOLD \
    --two_opt_max_iter $TWO_OPT_MAX_ITER --config_path "${CONFIG_PATH_ARGS[@]}";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo -e "${BLUE}Running without fast_tsp...${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    uv run python main.py test_sim --policies "${POLICIES[@]}" --data_distribution "$DATA_DIST" --dt "$DECODE_TYPE" \
    --cc "$n_cores" --n_samples "$N_SAMPLES" --area "$AREA" --bin_idx_file "$IDX_PATH" --size "$N_BINS" --seed "$SEED" \
    --problem "$PROBLEM" --n_vehicles "$VEHICLES" --vm "$VERTEX_METHOD" --lac "${LOOKAHEAD_CONFIGS[@]}" --dm "$DIST_METHOD" \
    --lvl "${REGULAR_LEVEL[@]}" --cf "${LAST_MINUTE_CF[@]}" --gp "${GUROBI_PARAM[@]}" --hp "${HEXALY_PARAM[@]}" \
    --et "$EDGE_THRESH" --em "$EDGE_METHOD" --waste_type "$WTYPE" --env_file "$ENV_FILE" --gplic_file "$GP_LIC_FILE" \
    --gapik_file "$GOOGLE_API_FILE" --waste_filepath "$WASTE_PATH" --symkey_name "$SYM_KEY" --dm_filepath "$DM_PATH" \
    --days "$N_DAYS" --cpd "$CHECKPOINTS" --stats_filepath "$STATS_PATH" --model_path "${MODEL_PATH_ARGS[@]}" \
    --gate_prob_threshold $GATE_PROB_THRESHOLD --mask_prob_threshold $MASK_PROB_THRESHOLD \
    --two_opt_max_iter $TWO_OPT_MAX_ITER --config_path "${CONFIG_PATH_ARGS[@]}";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
fi

echo ""
echo -e "${GREEN}✓ [SUCCESS] Simulation test completed successfully.${NC}"
