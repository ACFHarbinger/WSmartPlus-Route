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


# Load Task Config first to get general settings and PROBLEM definition
TASK_CONFIG="assets/configs/tasks/meta_train.yaml"
DATA_CONFIG="assets/configs/data/meta_train.yaml"
eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$TASK_CONFIG" "$DATA_CONFIG")

# Now load the specific environment config based on the problem defined in the task
if [ -n "$PROBLEM" ]; then
    ENV_CONFIG="assets/configs/envs/${PROBLEM}.yaml"
    if [ -f "$ENV_CONFIG" ]; then
        # Load Task + Data + Env (Task overrides Env values like Rewards for Meta-RL)
        eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$ENV_CONFIG" "$DATA_CONFIG" "$TASK_CONFIG")
    fi
fi

# Load Model Config (defaulting to the first model in the list for initial shell variables)
MODEL_NAME_0="${MODEL_NAMES[0]}"
MODEL_CONFIG="assets/configs/models/${MODEL_NAME_0}.yaml"
if [ -f "$MODEL_CONFIG" ]; then
    eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$ENV_CONFIG" "$DATA_CONFIG" "$MODEL_CONFIG" "$TASK_CONFIG")
fi

# MAP ENVIRONMENT VARIABLES TO SCRIPT VARIABLES
if [ -n "$VERTEX_M" ]; then VERTEX_M="$VERTEX_M"; fi
if [ -n "$DIST_M" ]; then DIST_M="$DIST_M"; fi
if [ -n "$EDGE_T" ]; then EDGE_T="$EDGE_T"; fi
if [ -n "$EDGE_M" ]; then EDGE_M="$EDGE_M"; fi

# Derived Variables
TOTAL_EPOCHS=$(($START + $EPOCHS))
DATASET_NAME="time${TOTAL_EPOCHS}"

DATASETS=()
# DATA_DISTS loaded from YAML
for dist in "${DATA_DISTS[@]}"; do
    DATASETS+=("data/datasets/${DATA_PROBLEM}/${DATA_PROBLEM}${SIZE}_${dist}_${DATASET_NAME}_seed${SEED}.pkl")
done


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

        MODEL_CONFIG="assets/configs/models/${M_NAME}.yaml"
        if [ -f "$MODEL_CONFIG" ]; then
            eval $(uv run python logic/src/utils/configs/yaml_to_env.py "$ENV_CONFIG" "$DATA_CONFIG" "$MODEL_CONFIG" "$TASK_CONFIG")
        fi

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
            model.n_heads="$N_HEADS" \
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
