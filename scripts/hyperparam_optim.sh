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

START=0
EPOCHS=7

EDGE_T=0.3
EDGE_M="knn"
VERTEX_M="mmn"
DATA_DIST="gamma1"

W_LEN=1.0
W_OVER=1.0
W_WASTE=1.0

EMBED_DIM=128
HIDDEN_DIM=512
N_ENC_L=3
N_ENC_SL=1
N_PRED_L=2
N_DEC_L=2

N_HEADS=8
NORM="instance"
ACTIVATION="gelu"
DROPOUT=0.1
AGG="mean"
AGG_G="mean"

OPTIM="rmsprop"
LR_MODEL=0.0001
LR_CV=0.0001
LR_SCHEDULER="lambda"
LR_DECAY=1.0

B_SIZE=256
N_DATA=128000
N_VAL_DATA=1280
VAL_B_SIZE=256

BL="exponential"
MAX_NORM=1.0
EXP_BETA=0.8
BL_ALPHA=0.05
ACC_STEPS=1

ETA=5
N_POP=20
FEVALS=25
METRIC="both"
HOP_METHOD="dehbo"
RANGE=(0.0 2.0)
MAX_TRES=40
H_EPOCHS=3
MUTPB=0.3
CXPB=0.5

SIZE=20
AREA="Rio Maior"
WTYPE="plastic"
F_SIZE=1
VAL_F_SIZE=1280
FOCUS_GRAPH="graphs_${SIZE}V_1N_${WTYPE}.json"

SEED=42
PROBLEM="wcvrp"
DATASET_NAME="real"
DATASET="data/datasets/${PROBLEM}/${PROBLEM}${SIZE}_${DATA_DIST}_${DATASET_NAME}_seed${SEED}.pkl"

TRAIN_AM=0
TRAIN_AMGC=1
TRAIN_TRANSGCN=1
TRAIN_DDAM=1
TRAIN_TAM=1
MODEL_NAMES=("am") #"amgc" "transgcn" "ddam" "tam")
MODEL_ENCODERS=("gat") #"gac" "tgc" "gat" "gat")
HORIZON=(0 0 0 0 3)
WB_MODE="disabled"

echo -e "${BLUE}Starting hyperparameter optimization module (Hydra-based)...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}$HOP_METHOD${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

# Multi-distribution loop (keeping structure from train.sh/meta_train.sh)
DATA_DISTS=("$DATA_DIST")
DATASETS=("$DATASET")

for dist_idx in "${!DATA_DISTS[@]}"; do
    D_DIST="${DATA_DISTS[dist_idx]}"
    T_DATASET="${DATASETS[dist_idx]}"

    echo -e "${BLUE}[PROCESS]${NC} Data distribution: ${YELLOW}${D_DIST}${NC}"

    for m_idx in "${!MODEL_NAMES[@]}"; do
        M_NAME="${MODEL_NAMES[m_idx]}"
        E_NAME="${MODEL_ENCODERS[m_idx]}"
        H_VAL="${HORIZON[m_idx]}"

        echo ""
        echo -e "${BLUE}===== [OPTIM] ${M_NAME} model with ${E_NAME} encoder =====${NC}"

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
            experiment=hpo \
            env.name="$PROBLEM" \
            model.name="$ACTUAL_MODEL_NAME" \
            model.encoder_type="$ACTUAL_ENCODER_TYPE" \
            hpo.n_epochs_per_trial="$H_EPOCHS" \
            env.waste_type="$WTYPE" \
            env.data_distribution="$D_DIST" \
            train.n_epochs="$EPOCHS" \
            env.vertex_method="$VERTEX_M" \
            hpo.eta="$ETA" \
            hpo.mutpb="$MUTPB" \
            train.batch_size="$B_SIZE" \
            train.train_data_size="$N_DATA" \
            train.val_data_size="$N_VAL_DATA" \
            model.temporal_horizon="${H_VAL}" \
            train.train_time=true \
            train.val_dataset="$T_DATASET" \
            model.normalization="$NORM" \
            model.embed_dim="$EMBED_DIM" \
            hpo.cxpb="$CXPB" \
            env.area="$AREA" \
            model.activation="$ACTIVATION" \
            model.num_encoder_layers="$N_ENC_L" \
            optim.optimizer="$OPTIM" \
            model.hidden_dim="$HIDDEN_DIM" \
            env.num_loc="$SIZE" \
            no_tensorboard=true \
            hpo.hop_range="[${RANGE[0]}, ${RANGE[1]}]" \
            hpo.fevals="$FEVALS" \
            hpo.metric="$METRIC" \
            hpo.n_pop="$N_POP" \
            env.focus_size="$F_SIZE" \
            env.eval_focus_size="$VAL_F_SIZE" \
            env.focus_graph="$FOCUS_GRAPH" \
            rl.exp_beta="$EXP_BETA" \
            optim.lr_scheduler="$LR_SCHEDULER" \
            optim.lr_decay="$LR_DECAY" \
            optim.lr="$LR_MODEL" \
            rl.lr_critic_value="$LR_CV" \
            model.dropout="$DROPOUT" \
            train.eval_batch_size="$VAL_B_SIZE" \
            model.aggregation_graph="$AGG_G" \
            rl.max_grad_norm="$MAX_NORM" \
            train.accumulation_steps="$ACC_STEPS" \
            seed="$SEED" \
            model.num_heads="$N_HEADS" \
            env.cost_weight="$W_LEN" \
            env.collection_reward="$W_WASTE" \
            env.overflow_penalty="$W_OVER" \
            train.epoch_start="$START" \
            wandb_mode="$WB_MODE" \
            $EXTRA_ARGS;

        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    done
done

echo ""
echo -e "${GREEN}✓ [DONE] Hyperparameter optimization completed.${NC}"
