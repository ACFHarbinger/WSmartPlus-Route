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


# Load configuration from YAML
CONFIG_FILE="scripts/configs/hyperparam_optim.yaml"

# Load variables
eval $(uv run python scripts/utils/yaml_to_env.py "$CONFIG_FILE")

# Derived Variables
# Construct DATASET path using loaded variables
DATASET="data/datasets/${PROBLEM}/${PROBLEM}${SIZE}_${DATA_DIST}_${DATASET_NAME}_seed${SEED}.pkl"

# Focus graph
FOCUS_GRAPH="graphs_${SIZE}V_1N_${WTYPE}.json"


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
            +experiment=hpo \
            env.name="'$PROBLEM'" \
            model.name="'$ACTUAL_MODEL_NAME'" \
            model.encoder_type="'$ACTUAL_ENCODER_TYPE'" \
            hpo.n_epochs_per_trial="$H_EPOCHS" \
            env.waste_type="'$WTYPE'" \
            env.data_distribution="'$D_DIST'" \
            train.n_epochs="$EPOCHS" \
            env.vertex_method="'$VERTEX_M'" \
            hpo.eta="$ETA" \
            hpo.mutpb="$MUTPB" \
            train.batch_size="$B_SIZE" \
            train.train_data_size="$N_DATA" \
            train.val_data_size="$N_VAL_DATA" \
            model.temporal_horizon="${H_VAL}" \
            train.train_time=true \
            train.val_dataset="'$T_DATASET'" \
            model.normalization="'$NORM'" \
            model.embed_dim="$EMBED_DIM" \
            hpo.cxpb="$CXPB" \
            env.area="'$AREA'" \
            model.activation="'$ACTIVATION'" \
            model.num_encoder_layers="$N_ENC_L" \
            optim.optimizer="'$OPTIM'" \
            model.hidden_dim="$HIDDEN_DIM" \
            env.num_loc="$SIZE" \
            no_tensorboard=true \
            hpo.hop_range="[${RANGE[0]}, ${RANGE[1]}]" \
            hpo.fevals="$FEVALS" \
            hpo.metric="'$METRIC'" \
            hpo.n_pop="$N_POP" \
            env.focus_size="$F_SIZE" \
            env.eval_focus_size="$VAL_F_SIZE" \
            env.focus_graph="'$FOCUS_GRAPH'" \
            rl.exp_beta="$EXP_BETA" \
            optim.lr_scheduler="'$LR_SCHEDULER'" \
            optim.lr_decay="$LR_DECAY" \
            optim.lr="$LR_MODEL" \
            rl.lr_critic_value="$LR_CV" \
            model.dropout="$DROPOUT" \
            train.eval_batch_size="$VAL_B_SIZE" \
            model.aggregation_graph="'$AGG_G'" \
            rl.max_grad_norm="$MAX_NORM" \
            train.accumulation_steps="$ACC_STEPS" \
            seed="$SEED" \
            model.num_heads="$N_HEADS" \
            env.cost_weight="$W_LEN" \
            env.collection_reward="$W_WASTE" \
            env.overflow_penalty="$W_OVER" \
            train.epoch_start="$START" \
            wandb_mode="'$WB_MODE'" \
            $EXTRA_ARGS;

        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    done
done

echo ""
echo -e "${GREEN}✓ [DONE] Hyperparameter optimization completed.${NC}"
