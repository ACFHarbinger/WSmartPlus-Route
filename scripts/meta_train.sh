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

EDGE_T=1.0
EDGE_M="knn"
DIST_M="gmaps"
VERTEX_M="mmn"

W_LEN=1.0
W_OVER=1000.0
W_WASTE=100.0

EMBED_DIM=128
HIDDEN_DIM=512
N_ENC_L=3
N_ENC_SL=1
N_PRED_L=2
N_DEC_L=2

N_HEADS=8
NORM="instance"
ACTI_F="gelu"
DROPOUT=0.1
AGG="sum"
AGG_G="avg"

OPTIM="rmsprop"
LR_MODEL=0.00005
LR_CV=0.0001
LR_SCHEDULER="lambda"
LR_DECAY=1.0

B_SIZE=256
N_DATA=1280
N_VAL_DATA=0
VAL_B_SIZE=0

BL="exponential"
MAX_NORM=1.0
EXP_BETA=0.8
BL_ALPHA=0.05
ACC_STEPS=1

GAT_HIDDEN=128
LSTM_HIDDEN=64
GATE_THRESH=0.5
META_METHOD="hrl"
META_HISTORY=10
META_LR=0.000005
META_STEP=10
META_B_SIZE=256
HRL_EPOCHS=2
HRL_CLIP_EPS=0.2
HRL_THRESHOLD=0.9

# HRL Hyperparameters for Rio Maior (Targeting < 1 overflow)
HRL_PID_TARGET=0.0003
HRL_LAMBDA_WASTE=300.0
HRL_LAMBDA_COST=0.5
HRL_LAMBDA_OVERFLOW_INIT=2000.0
HRL_LAMBDA_OVERFLOW_MIN=100.0
HRL_LAMBDA_OVERFLOW_MAX=5000.0
HRL_LAMBDA_PRUNING=0.5
HRL_LAMBDA_MASK_AUX=5.0
HRL_ENTROPY=0.01

SIZE=100
LOG_STEP=10
AREA="riomaior"
WTYPE="plastic"
F_SIZE=1280
VAL_F_SIZE=0
DM_METHOD="gmaps"
F_GRAPH="graphs_${SIZE}V_1N_${WTYPE}.json"
DM_PATH="data/wsr_simulator/distance_matrix/gmaps_distmat_${WTYPE}[${AREA}].csv"

SEED=42
START=0
EPOCHS=100
TOTAL_EPOCHS=$(($START + $EPOCHS))
PROBLEM="cwcvrp"
DATA_PROBLEM="wcvrp"
DATASET_NAME="time${TOTAL_EPOCHS}"
DATASETS=()
DATA_DISTS=("gamma1")
for dist in "${DATA_DISTS[@]}"; do
    DATASETS+=("data/datasets/${DATA_PROBLEM}/${DATA_PROBLEM}${SIZE}_${dist}_${DATASET_NAME}_seed${SEED}.pkl")
done

MODEL_NAMES=("am") #"amgc" "transgcn" "ddam" "tam")
MODEL_ENCODERS=("gat") #"gac" "tgc" "gat" "gat")
HORIZON=(0 0 0 0 3)
WB_MODE="disabled"
LOAD_PATH=""

echo -e "${BLUE}Starting Meta-RL Training Module (Hydra-based)...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Epochs:     ${MAGENTA}$EPOCHS${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}$META_METHOD${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

for dist_idx in "${!DATA_DISTS[@]}"; do
    DATA_DIST="${DATA_DISTS[dist_idx]}"
    TRAIN_DATASET="${DATASETS[dist_idx]}"

    echo -e "${BLUE}[PROCESS]${NC} Meta-RL Data Distribution: ${YELLOW}${DATA_DIST}${NC}"

    for m_idx in "${!MODEL_NAMES[@]}"; do
        M_NAME="${MODEL_NAMES[m_idx]}"
        E_NAME="${MODEL_ENCODERS[m_idx]}"
        H_VAL="${HORIZON[m_idx]}"

        echo ""
        echo -e "${BLUE}===== [TRAIN] ${M_NAME} model (HRL) with ${E_NAME} encoder =====${NC}"

        # Determine actual Hydra parameters based on model type
        ACTUAL_MODEL_NAME="$M_NAME"
        ACTUAL_ENCODER_TYPE="$E_NAME"
        EXTRA_ARGS=""

        case "$M_NAME" in
            "amgc")
                ACTUAL_MODEL_NAME="am"
                ACTUAL_ENCODER_TYPE="ggac"
                ;;
            "transgcn")
                ACTUAL_MODEL_NAME="am"
                ACTUAL_ENCODER_TYPE="tgc"
                EXTRA_ARGS="model.num_encoder_sublayers=${N_ENC_SL}"
                ;;
            "ddam")
                ACTUAL_MODEL_NAME="ddam"
                ACTUAL_ENCODER_TYPE="gat"
                EXTRA_ARGS="model.num_decoder_layers=${N_DEC_L}"
                ;;
            "tam")
                ACTUAL_MODEL_NAME="tam"
                ACTUAL_ENCODER_TYPE="gat"
                EXTRA_ARGS="model.num_predictor_layers=${N_PRED_L}"
                ;;
        esac

        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi

        uv run python main.py train_lightning \
            experiment=meta_rl \
            env.name="'$PROBLEM'" \
            model.name="'$ACTUAL_MODEL_NAME'" \
            model.encoder_type="'$ACTUAL_ENCODER_TYPE'" \
            train.train_data_size="$N_DATA" \
            env.data_distribution="'${DATA_DIST}'" \
            env.num_loc="$SIZE" \
            train.n_epochs="$EPOCHS" \
            seed="$SEED" \
            train.train_time=true \
            env.vertex_method="'$VERTEX_M'" \
            train.epoch_start="$START" \
            rl.max_grad_norm="$MAX_NORM" \
            train.val_data_size="$N_VAL_DATA" \
            env.cost_weight="$W_LEN" \
            env.collection_reward="$W_WASTE" \
            env.overflow_penalty="$W_OVER" \
            model.embed_dim="$EMBED_DIM" \
            model.activation="'$ACTI_F'" \
            train.accumulation_steps="$ACC_STEPS" \
            env.focus_graph="'$F_GRAPH'" \
            model.normalization="'$NORM'" \
            train.log_step="$LOG_STEP" \
            optim.optimizer="'$OPTIM'" \
            model.hidden_dim="$HIDDEN_DIM" \
            model.num_heads="$N_HEADS" \
            model.dropout="$DROPOUT" \
            env.waste_type="'$WTYPE'" \
            env.focus_size="$F_SIZE" \
            model.num_encoder_layers="$N_ENC_L" \
            optim.lr="$LR_MODEL" \
            env.eval_focus_size="$VAL_F_SIZE" \
            env.distance_method="'$DIST_M'" \
            rl.exp_beta="$EXP_BETA" \
            rl.hrl_threshold="$HRL_THRESHOLD" \
            model.temporal_horizon="${H_VAL}" \
            optim.lr_scheduler="'$LR_SCHEDULER'" \
            optim.lr_decay="$LR_DECAY" \
            train.batch_size="$B_SIZE" \
            rl.lr_critic_value="$LR_CV" \
            rl.bl_alpha="$BL_ALPHA" \
            env.area="'$AREA'" \
            model.aggregation_node="'$AGG'" \
            model.aggregation_graph="'$AGG_G'" \
            env.dm_filepath="'$DM_PATH'" \
            env.edge_threshold="$EDGE_T" \
            env.edge_method="'$EDGE_M'" \
            train.eval_batch_size="$VAL_B_SIZE" \
            rl.mrl_batch_size="$META_B_SIZE" \
            wandb_mode="'$WB_MODE'" \
            rl.meta_strategy="'$META_METHOD'" \
            rl.meta_lr="$META_LR" \
            rl.meta_history_length="$META_HISTORY" \
            rl.mrl_step="$META_STEP" \
            rl.hrl_epochs="$HRL_EPOCHS" \
            rl.hrl_clip_eps="$HRL_CLIP_EPS" \
            rl.gat_hidden_dim="$GAT_HIDDEN" \
            rl.lstm_hidden_dim="$LSTM_HIDDEN" \
            rl.gate_prob_threshold="$GATE_THRESH" \
            train.load_path="'$LOAD_PATH'" \
            rl.shared_encoder=true \
            rl.hrl_pid_target="$HRL_PID_TARGET" \
            rl.hrl_lambda_waste="$HRL_LAMBDA_WASTE" \
            rl.hrl_lambda_cost="$HRL_LAMBDA_COST" \
            rl.hrl_lambda_overflow_initial="$HRL_LAMBDA_OVERFLOW_INIT" \
            rl.hrl_lambda_overflow_min="$HRL_LAMBDA_OVERFLOW_MIN" \
            rl.hrl_lambda_overflow_max="$HRL_LAMBDA_OVERFLOW_MAX" \
            rl.hrl_lambda_pruning="$HRL_LAMBDA_PRUNING" \
            rl.hrl_lambda_mask_aux="$HRL_LAMBDA_MASK_AUX" \
            rl.entropy_weight="$HRL_ENTROPY" \
            rl.baseline="'$BL'" \
            $EXTRA_ARGS;

        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    done
done

echo ""
echo -e "${GREEN}✓ [DONE] Meta-RL Training completed for all distributions.${NC}"
