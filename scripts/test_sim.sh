#!/bin/bash

# Default to quiet mode
VERBOSE=false

# Default cores
N_CORES=1

while getopts nc: flag
do
    case "${flag}" in
        nc) n_cores=${OPTARG};;
    esac
done

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

if [[ -z $n_cores ]]; then
    n_cores=$N_CORES
fi

SEED=42
N_DAYS=31
N_BINS=100
N_SAMPLES=1
PROBLEM="vrpp"

AREA="riomaior"
WTYPE="plastic"
DATA_DIST="emp"
IDX_PATH="graphs_${N_BINS}V_1N_${WTYPE}.json"

SYM_KEY="skey"
ENV_FILE="vars.env"
GP_LIC_FILE="gurobi.lic.enc"
HEX_DAT_FILE="hexaly.dat.enc"
GOOGLE_API_FILE="google.lic.enc"

REGULAR_LEVEL=(3)
LAST_MINUTE_CF=(90)
GUROBI_PARAM=(0.84)
HEXALY_PARAM=(0.84)
DECODE_TYPE="greedy"
LOOKAHEAD_CONFIGS=('a') #'a' 'b'
POLICIES=("policy_look_ahead_alns")
#"policy_look_ahead" "policy_look_ahead_vrpp" "policy_look_ahead_sans" 
#"policy_look_ahead_hgs" "policy_look_ahead_alns" "policy_look_ahead_bcp"
#"policy_last_minute_and_path" "policy_last_minute" "policy_regular" 
#"gurobi_vrpp" "hexaly_vrpp" 
#"am" "amgc" "transgcn"

VEHICLES=0
EDGE_THRESH=0.0
EDGE_METHOD="knn"
VERTEX_METHOD="mmn"
DIST_METHOD="gmaps"
DM_PATH="data/wsr_simulator/distance_matrix/gmaps_distmat_${WTYPE}[${AREA}].csv"
WASTE_PATH="daily_waste/${AREA}${N_BINS}_${DATA_DIST}_wsr31_N1_seed${SEED}.pkl"

RUN_TSP=1
CHECKPOINTS=30

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
    --lac "${LOOKAHEAD_CONFIGS[@]}" --hp "$HEXALY_PARAM" --problem "$PROBLEM" --days "$N_DAYS" --dm "$DIST_METHOD" \
    --waste_type "$WTYPE" --cc "$n_cores" --et "$EDGE_THRESH" --em "$EDGE_METHOD" --env_file "$ENV_FILE" \
    --gapik_file "$GOOGLE_API_FILE" --symkey_name "$SYM_KEY" --dm_filepath "$DM_PATH" --waste_filepath "$WASTE_PATH";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo "Running without fast_tsp..."
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py test_sim --policies "${POLICIES[@]}" --data_distribution "$DATA_DIST" --dt "$DECODE_TYPE" --cpd "$CHECKPOINTS" \
    --cc "$n_cores" --n_samples "$N_SAMPLES" --area "$AREA" --bin_idx_file "$IDX_PATH" --size "$N_BINS" --seed "$SEED" \
    --problem "$PROBLEM" --n_vehicles "$VEHICLES" --vm "$VERTEX_METHOD" --lac "${LOOKAHEAD_CONFIGS[@]}" --dm "$DIST_METHOD" \
    --days "$N_DAYS" --lvl "${REGULAR_LEVEL[@]}" --cf "${LAST_MINUTE_CF[@]}" --gp "${GUROBI_PARAM[@]}" --hp "$HEXALY_PARAM" \
    --et "$EDGE_THRESH" --em "$EDGE_METHOD" --waste_type "$WTYPE" --env_file "$ENV_FILE" --gplic_file "$GP_LIC_FILE" \
    --gapik_file "$GOOGLE_API_FILE" --waste_filepath "$WASTE_PATH" --symkey_name "$SYM_KEY" --dm_filepath "$DM_PATH";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
fi

echo ""
echo "========================================"
echo "Test completed successfully"
echo "========================================"
