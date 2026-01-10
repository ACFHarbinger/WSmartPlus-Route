#!/bin/bash

# Default to quiet mode
VERBOSE=false

# Default cores
N_CORES=22

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
        F) RUN_TSP=1;;
        v) VERBOSE=true;;
        C) CONFIG_PATH=${OPTARG};;
    esac
done

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2  # Save original stdout (fd1) to fd3, stderr (fd2) to fd4
    exec >/dev/null 2>&1
fi

if [[ -z $n_cores ]]; then
    n_cores=$N_CORES
fi

SEED=42
N_DAYS=31
N_BINS=100
N_SAMPLES=1
PROBLEM="cwcvrp"

AREA="riomaior"
WTYPE="plastic"
DATA_DIST="gamma1"
IDX_PATH="graphs_${N_BINS}V_1N_${WTYPE}.json"
STATS_PATH="" #"daily_waste/april_2024_summary.csv"

SYM_KEY="skey"
ENV_FILE="vars.env"
GP_LIC_FILE="gurobi.lic"
HEX_DAT_FILE="hexaly.dat.enc"
GOOGLE_API_FILE="google.lic.enc"

REGULAR_LEVEL=(3 4)
LAST_MINUTE_CF=(70)
GUROBI_PARAM=(0.84)
HEXALY_PARAM=(0.84)
DECODE_TYPE="greedy"
LOOKAHEAD_CONFIGS=('a') #'a' 'b'
POLICIES=("policy_look_ahead_hgs")
#"policy_look_ahead" "policy_look_ahead_vrpp" "policy_look_ahead_sans" 
#"policy_look_ahead_hgs" "policy_look_ahead_alns" "policy_look_ahead_bcp"
#"policy_last_minute_and_path" "policy_last_minute" "policy_regular" 
#"gurobi_vrpp" "hexaly_vrpp" 
#"amgat" "amggac" "amtgc"
declare -A MODEL_PATHS
MODEL_PATHS["amgat"]="${PROBLEM}${N_BINS}_${AREA}_${WTYPE}/${DATA_DIST}/amgat"
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

declare -A CONFIG_PATHS
CONFIG_PATHS["hgs"]="assets/configs/lookahead_hgs.yaml"
CONFIG_PATHS["alns"]="assets/configs/lookahead_alns.yaml"
CONFIG_PATHS["sans"]="assets/configs/lookahead_sans.yaml"
CONFIG_PATHS["bcp"]="assets/configs/lookahead_bcp.yaml"
CONFIG_PATHS["vrpp"]="assets/configs/vrpp.yaml"
CONFIG_PATHS["lookahead_a"]="assets/configs/lookahead_a.yaml"
CONFIG_PATHS["lookahead_b"]="assets/configs/lookahead_b.yaml"

CONFIG_PATH_ARGS=()
for key in "${!CONFIG_PATHS[@]}"; do
    CONFIG_PATH_ARGS+=("$key=${CONFIG_PATHS[$key]}")
done

if [[ -n "$CONFIG_PATH" ]]; then
    # Add manual config path to the list (user override)
    CONFIG_PATH_ARGS+=("$CONFIG_PATH")
fi

VEHICLES=0
REAL_TIME_LOG=1
GATE_PROB_THRESHOLD="0.1"
MASK_PROB_THRESHOLD="${MASK_PROB_THRESHOLD:-0.1}"

EDGE_THRESH=1.0
EDGE_METHOD="knn"
VERTEX_METHOD="mmn"
DIST_METHOD="gmaps"

# Update dependent paths based on parsed arguments
IDX_PATH="graphs_${N_BINS}V_1N_${WTYPE}.json"
DM_PATH="data/wsr_simulator/distance_matrix/gmaps_distmat_${WTYPE}[${AREA}].csv"
WASTE_PATH=""
CHECKPOINTS=30

# Route postprocessing
RUN_TSP=1
TWO_OPT_MAX_ITER=0

echo "Starting test execution with $n_cores cores..."
echo "========================================"
echo "Test Configuration"
echo "========================================"
echo "Cores: $n_cores"
echo "Problem: $PROBLEM"
echo "Area: $AREA"
echo "Policies: $POLICIES"
echo "Samples: $N_SAMPLES"
echo "Days: $N_DAYS"
echo "Fast TSP Mode: $RUN_TSP"
echo "========================================"
echo ""

# Add option --real_time_log to the command line if you want real time updates of the simulation
if [ "$RUN_TSP" -eq 0 ]; then
    echo "Running with fast_tsp..."
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py test_sim --policies "${POLICIES[@]}" --data_distribution "$DATA_DIST" --run_tsp \
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
    echo "Running without fast_tsp..."
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    ./.venv/bin/python3 main.py test_sim --policies "${POLICIES[@]}" --data_distribution "$DATA_DIST" --dt "$DECODE_TYPE" \
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
echo "========================================"
echo "Test completed successfully"
echo "========================================"
