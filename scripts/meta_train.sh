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
# emp W_LEN = 1.5, 1.0, 1.0, 1.0
# gamma W_LEN = 2.5, 1.75, 1.75, 1.75

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
N_VAL_DATA=0 #1280
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
# HRL lambda limits
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

# Consistency variables
N_BINS=$SIZE
N_DAYS=31
N_SAMPLES=1
DATA_DIST="gamma1"
WASTE_PATH="daily_waste/${AREA}${N_BINS}_${DATA_DIST}_wsr${N_DAYS}_N${N_SAMPLES}_seed${SEED}.pkl"

SEED=42
START=0
EPOCHS=100
TOTAL_EPOCHS=$(($START + $EPOCHS))
PROBLEM="cwcvrp"
DATA_PROBLEM="wcvrp"
DATASET_NAME="time${TOTAL_EPOCHS}"
VAL_DATASET_NAME="${DATASET_NAME}_val"
DATASETS=()
VAL_DATASETS=()
DATA_DISTS=("gamma1")
for dist in "${DATA_DISTS[@]}"; do
    # VAL_DATASETS+="data/datasets/${DATA_PROBLEM}/${DATA_PROBLEM}${SIZE}_${dist}_${VAL_DATASET_NAME}_seed${SEED}.pkl"
    DATASETS+=("data/datasets/${DATA_PROBLEM}/${DATA_PROBLEM}${SIZE}_${dist}_${DATASET_NAME}_seed${SEED}.pkl")
done

TRAIN_AM=0
TRAIN_AMGC=1
TRAIN_TRANSGCN=1
TRAIN_DDAM=1
TRAIN_TAM=1
HORIZON=(0 0 0 0 3)
WB_MODE="disabled" # 'online'|'offline'|'disabled'
# LOAD_PATH is empty for training from scratch
LOAD_PATH=""

echo -e "${BLUE}Starting Meta-RL Training Module...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Epochs:     ${MAGENTA}$EPOCHS${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}$META_METHOD${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

for ((id = 0; id < ${#DATA_DISTS[@]}; id++)); do
    echo -e "${BLUE}[PROCESS]${NC} Meta-RL Data Distribution: ${YELLOW}${DATA_DISTS[id]}${NC}"
    if [ "$TRAIN_AM" -eq 0 ]; then
        echo -e "${BLUE}===== [TRAIN] AM model (HRL) =====${NC}"
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py mrl_train --problem "$PROBLEM" --model am --encoder gat --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --log_step "$LOG_STEP" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" --hrl_threshold "$HRL_THRESHOLD" \
        --temporal_horizon "${HORIZON[0]}" --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" \
        --batch_size "$B_SIZE" --lr_critic_value "$LR_CV" --bl_alpha "$BL_ALPHA" --area "$AREA" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" --mrl_batch_size "$META_B_SIZE" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --mrl_method "$META_METHOD" --mrl_lr "$META_LR" \
        --mrl_history "$META_HISTORY" --mrl_step "$META_STEP" --hrl_epochs "$HRL_EPOCHS" --hrl_clip_eps "$HRL_CLIP_EPS" \
        --gat_hidden "$GAT_HIDDEN" --lstm_hidden "$LSTM_HIDDEN" --gate_prob_threshold "$GATE_THRESH" --load_path "$LOAD_PATH" \
        --shared_encoder --hrl_pid_target "$HRL_PID_TARGET" --hrl_lambda_waste "$HRL_LAMBDA_WASTE" \
        --hrl_lambda_cost "$HRL_LAMBDA_COST" --hrl_lambda_overflow_initial "$HRL_LAMBDA_OVERFLOW_INIT" \
        --hrl_lambda_overflow_min "$HRL_LAMBDA_OVERFLOW_MIN" --hrl_lambda_overflow_max "$HRL_LAMBDA_OVERFLOW_MAX" \
        --hrl_lambda_pruning "$HRL_LAMBDA_PRUNING" \
        --hrl_lambda_mask_aux "$HRL_LAMBDA_MASK_AUX" --hrl_entropy_coef "$HRL_ENTROPY" --aggregation "$AGG" \
        --baseline "$BL";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Skipping AM HRL training (TRAIN_AM=$TRAIN_AM)"
    fi
    if [ "$TRAIN_AMGC" -eq 0 ]; then
        echo ""
        echo -e "${BLUE}===== [TRAIN] AMGC model (HRL) =====${NC}"
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py mrl_train --problem "$PROBLEM" --model am --encoder gac --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --log_step "$LOG_STEP" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" --hrl_threshold "$HRL_THRESHOLD" \
        --temporal_horizon "${HORIZON[1]}" --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY"  \
        --batch_size "$B_SIZE" --lr_critic_value "$LR_CV" --bl_alpha "$BL_ALPHA" --area "$AREA" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --mrl_method "$META_METHOD" --mrl_lr "$META_LR" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" --mrl_batch_size "$META_B_SIZE" \
        --mrl_history "$META_HISTORY" --mrl_step "$META_STEP" --hrl_epochs "$HRL_EPOCHS" --hrl_clip_eps "$HRL_CLIP_EPS" \
        --gat_hidden "$GAT_HIDDEN" --lstm_hidden "$LSTM_HIDDEN" --gate_prob_threshold "$GATE_THRESH" \
        --shared_encoder --hrl_pid_target "$HRL_PID_TARGET" --hrl_lambda_waste "$HRL_LAMBDA_WASTE" \
        --hrl_lambda_cost "$HRL_LAMBDA_COST" --hrl_lambda_overflow_initial "$HRL_LAMBDA_OVERFLOW_INIT" \
        --hrl_lambda_overflow_min "$HRL_LAMBDA_OVERFLOW_MIN" --hrl_lambda_overflow_max "$HRL_LAMBDA_OVERFLOW_MAX" \
        --hrl_lambda_pruning "$HRL_LAMBDA_PRUNING" --hrl_lambda_mask_aux "$HRL_LAMBDA_MASK_AUX" \
        --hrl_entropy_coef "$HRL_ENTROPY" --aggregation "$AGG" \
        --baseline "$BL";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Skipping AMGC HRL training (TRAIN_AMGC=$TRAIN_AMGC)"
    fi

    if [ "$TRAIN_TRANSGCN" -eq 0 ]; then
        echo ""
        echo -e "${BLUE}===== [TRAIN] TRANSGCN model (HRL) =====${NC}"
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py mrl_train --problem "$PROBLEM" --model am --encoder tgc --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --hrl_threshold "$HRL_THRESHOLD" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --log_step "$LOG_STEP" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" --area "$AREA" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" --seed "$SEED" \
        --temporal_horizon "${HORIZON[2]}" --lr_scheduler "$LR_SCHEDULER" --n_encode_sublayers "$N_ENC_SL" \
        --batch_size "$B_SIZE" --lr_critic_value "$LR_CV" --bl_alpha "$BL_ALPHA" --lr_decay "$LR_DECAY" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --mrl_method "$META_METHOD" --mrl_lr "$META_LR" \
        --mrl_history "$META_HISTORY" --mrl_step "$META_STEP" --hrl_epochs "$HRL_EPOCHS" --hrl_clip_eps "$HRL_CLIP_EPS" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" --mrl_batch_size "$META_B_SIZE" \
        --gat_hidden "$GAT_HIDDEN" --lstm_hidden "$LSTM_HIDDEN" --gate_prob_threshold "$GATE_THRESH" \
        --shared_encoder --hrl_pid_target "$HRL_PID_TARGET" --hrl_lambda_waste "$HRL_LAMBDA_WASTE" \
        --hrl_lambda_cost "$HRL_LAMBDA_COST" --hrl_lambda_overflow_initial "$HRL_LAMBDA_OVERFLOW_INIT" \
        --hrl_lambda_overflow_min "$HRL_LAMBDA_OVERFLOW_MIN" --hrl_lambda_overflow_max "$HRL_LAMBDA_OVERFLOW_MAX" \
        --hrl_lambda_pruning "$HRL_LAMBDA_PRUNING" \
        --hrl_lambda_mask_aux "$HRL_LAMBDA_MASK_AUX" --hrl_entropy_coef "$HRL_ENTROPY" --aggregation "$AGG" \
        --baseline "$BL";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Skipping TRANSGCN HRL training (TRAIN_TRANSGCN=$TRAIN_TRANSGCN)"
    fi

    if [ "$TRAIN_DDAM" -eq 0 ]; then
        echo ""
        echo -e "${BLUE}===== [TRAIN] DDAM model (HRL) =====${NC}"
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py mrl_train --problem "$PROBLEM" --model ddam --encoder gat --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" --hrl_threshold "$HRL_THRESHOLD" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --area "$AREA" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" --log_step "$LOG_STEP" \
        --temporal_horizon "${HORIZON[3]}" --lr_scheduler "$LR_SCHEDULER" --n_decode_layers "$N_DEC_L"  \
        --batch_size "$B_SIZE" --lr_critic_value "$LR_CV" --bl_alpha "$BL_ALPHA" --lr_decay "$LR_DECAY" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --mrl_method "$META_METHOD" --mrl_lr "$META_LR" \
        --mrl_history "$META_HISTORY" --mrl_step "$META_STEP" --hrl_epochs "$HRL_EPOCHS" --hrl_clip_eps "$HRL_CLIP_EPS" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" --mrl_batch_size "$META_B_SIZE" \
        --gat_hidden "$GAT_HIDDEN" --lstm_hidden "$LSTM_HIDDEN" --gate_prob_threshold "$GATE_THRESH" \
        --shared_encoder --hrl_pid_target "$HRL_PID_TARGET" --hrl_lambda_waste "$HRL_LAMBDA_WASTE" \
        --hrl_lambda_cost "$HRL_LAMBDA_COST" --hrl_lambda_overflow_initial "$HRL_LAMBDA_OVERFLOW_INIT" \
        --hrl_lambda_overflow_min "$HRL_LAMBDA_OVERFLOW_MIN" --hrl_lambda_overflow_max "$HRL_LAMBDA_OVERFLOW_MAX" \
        --hrl_lambda_pruning "$HRL_LAMBDA_PRUNING" \
        --hrl_lambda_mask_aux "$HRL_LAMBDA_MASK_AUX" --hrl_entropy_coef "$HRL_ENTROPY" --aggregation "$AGG" \
        --baseline "$BL";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Skipping DDAM HRL training (TRAIN_DDAM=$TRAIN_DDAM)"
    fi

    if [ "$TRAIN_TAM" -eq 0 ]; then
        echo ""
        echo -e "${BLUE}===== [TRAIN] TAM model (HRL) =====${NC}"
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py mrl_train --problem "$PROBLEM" --model tam --encoder gat --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" --hrl_threshold "$HRL_THRESHOLD" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --area "$AREA" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" --log_step "$LOG_STEP" \
        --temporal_horizon "${HORIZON[4]}" --lr_scheduler "$LR_SCHEDULER" --n_predict_layers "$N_PRED_L"  \
        --batch_size "$B_SIZE" --lr_critic_value "$LR_CV" --bl_alpha "$BL_ALPHA" --lr_decay "$LR_DECAY" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --mrl_method "$META_METHOD" --mrl_lr "$META_LR" \
        --mrl_history "$META_HISTORY" --mrl_step "$META_STEP" --hrl_epochs "$HRL_EPOCHS" --hrl_clip_eps "$HRL_CLIP_EPS" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" --mrl_batch_size "$META_B_SIZE" \
        --gat_hidden "$GAT_HIDDEN" --lstm_hidden "$LSTM_HIDDEN" --gate_prob_threshold "$GATE_THRESH" \
        --shared_encoder --hrl_pid_target "$HRL_PID_TARGET" --hrl_lambda_waste "$HRL_LAMBDA_WASTE" \
        --hrl_lambda_cost "$HRL_LAMBDA_COST" --hrl_lambda_overflow_initial "$HRL_LAMBDA_OVERFLOW_INIT" \
        --hrl_lambda_overflow_min "$HRL_LAMBDA_OVERFLOW_MIN" --hrl_lambda_overflow_max "$HRL_LAMBDA_OVERFLOW_MAX" \
        --hrl_lambda_pruning "$HRL_LAMBDA_PRUNING" \
        --hrl_lambda_mask_aux "$HRL_LAMBDA_MASK_AUX" --hrl_entropy_coef "$HRL_ENTROPY" --aggregation "$AGG" \
        --baseline "$BL";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo -e "${YELLOW}[SKIP]${NC} Skipping TAM HRL training (TRAIN_TAM=$TRAIN_TAM)"
    fi
done

echo ""
echo -e "${GREEN}✓ [DONE] Meta-RL Training completed for all distributions.${NC}"