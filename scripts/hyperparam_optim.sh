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
HOP_METHOD="dehb"
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
VAL_DATASET="${DATASET_NAME}_val"
DATA_DIST="gamma1"
# VAL_DATASET="data/datasets/${PROBLEM}/${PROBLEM}${SIZE}_${DATA_DIST}_${VAL_DATASET}_seed${SEED}.pkl"
DATASET="data/datasets/${PROBLEM}/${PROBLEM}${SIZE}_${DATA_DIST}_${DATASET_NAME}_seed${SEED}.pkl"

TRAIN_AM=0
TRAIN_AMGC=1
TRAIN_TRANSGCN=1
TRAIN_DDAM=1
TRAIN_TAM=1
HORIZON=(0 0 0 0 3)
WB_MODE="disabled"

echo -e "${BLUE}Starting hyperparameter optimization module...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}$HOP_METHOD${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

if [ "$TRAIN_AM" -eq 0 ]; then
    echo -e "${BLUE}===== [OPTIM] AM model =====${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py hp_optim --problem "$PROBLEM" --model am --encoder gat --hop_epochs "$H_EPOCHS" --waste_type "$WTYPE" \
    --data_distribution "$DATA_DIST" --n_epochs "$EPOCHS" --vertex_method "$VERTEX_M" --eta "$ETA" --mutpb "$MUTPB" \
    --batch_size "$B_SIZE" --epoch_size "$N_DATA" --val_size "$N_VAL_DATA" --temporal_horizon "${HORIZON[0]}" \
    --train_time --train_dataset "$DATASET" --normalization "$NORM"  --embedding_dim "$EMBED_DIM" --cxpb "$CXPB" \
    --area "$AREA" --activation "$ACTIVATION" --n_encode_layers "$N_ENC_L" --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" \
    --graph_size "$SIZE" --no_tensorboard --range "${RANGE[@]}" --fevals "$FEVALS" --metric "$METRIC" --n_pop "$N_POP" \
    --focus_size "$F_SIZE" --eval_focus_size "$VAL_F_SIZE" --focus_graph "$FOCUS_GRAPH" --exp_beta "$EXP_BETA" \
    --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" --lr_model "$LR_MODEL" --lr_critic_value "$LR_CV" --dropout "$DROPOUT" \
    --eval_batch_size "$VAL_B_SIZE" --aggregation_graph "$AGG_G" --max_grad_norm "$MAX_NORM" --accumulation_steps "$ACC_STEPS" \
    --seed "$SEED" --n_heads "$N_HEADS" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" --epoch_start "$START" \
    --wandb_mode "$WB_MODE";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Skipping AM optimization (TRAIN_AM=$TRAIN_AM)"
fi

if [ "$TRAIN_AMGC" -eq 0 ]; then
    echo ""
    echo -e "${BLUE}===== [OPTIM] AMGC model =====${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py hp_optim --problem "$PROBLEM" --model am --encoder gac --hop_epochs "$H_EPOCHS" --waste_type "$WTYPE" \
    --data_distribution "$DATA_DIST" --n_epochs "$EPOCHS" --vertex_method "$VERTEX_M" --mutpb "$MUTPB" --cxpb "$CXPB" \
    --batch_size "$B_SIZE" --epoch_size "$N_DATA" --val_size "$N_VAL_DATA" --n_pop "$N_POP" --edge_method "$EDGE_M" --eta "$ETA" \
    --train_dataset "$DATASET" --normalization "$NORM" --embedding_dim "$EMBED_DIM" --edge_threshold "$EDGE_T" --area "$AREA" \
    --train_time --activation "$ACTIVATION" --n_encode_layers "$N_ENC_L" --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" \
    --no_tensorboard --graph_size "$SIZE" --range "${RANGE[@]}" --fevals "$FEVALS" --metric "$METRIC" --aggregation "$AGG" \
    --focus_size "$F_SIZE" --eval_focus_size "$VAL_F_SIZE" --focus_graph "$FOCUS_GRAPH" --temporal_horizon "${HORIZON[1]}" \
    --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" --lr_model "$LR_MODEL" --lr_critic_value "$LR_CV" --dropout "$DROPOUT" \
    --eval_batch_size "$VAL_B_SIZE" --aggregation_graph "$AGG_G" --max_grad_norm "$MAX_NORM" --accumulation_steps "$ACC_STEPS" \
    --seed "$SEED" --n_heads "$N_HEADS" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" --epoch_start "$START" \
    --wandb_mode "$WB_MODE";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Skipping AMGC optimization (TRAIN_AMGC=$TRAIN_AMGC)"
fi

if [ "$TRAIN_TRANSGCN" -eq 0 ]; then
    echo ""
    echo -e "${BLUE}===== [OPTIM] TRANSGCN model =====${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py hp_optim --problem "$PROBLEM" --model am --encoder tgc --hop_epochs "$H_EPOCHS" --waste_type "$WTYPE" \
    --data_distribution "$DATA_DIST" --n_epochs "$EPOCHS" --vertex_method "$VERTEX_M" --eta "$ETA" --mutpb "$MUTPB" \
    --batch_size "$B_SIZE" --epoch_size "$N_DATA" --val_size "$N_VAL_DATA" --n_pop "$N_POP" --area "$AREA" \
    --train_time --train_dataset "$DATASET" --normalization "$NORM" --embedding_dim "$EMBED_DIM" --aggregation "$AGG" \
    --activation "$ACTIVATION" --n_encode_layers "$N_ENC_L" --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" \
    --graph_size "$SIZE" --no_tensorboard --range "${RANGE[@]}" --fevals "$FEVALS" --metric "$METRIC" --cxpb "$CXPB" \
    --focus_size "$F_SIZE" --eval_focus_size "$VAL_F_SIZE" --focus_graph "$FOCUS_GRAPH" --temporal_horizon "${HORIZON[2]}" \
    --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" --lr_model "$LR_MODEL" --lr_critic_value "$LR_CV" \
    --n_encode_sublayers "$N_ENC_SL" --eval_batch_size "$VAL_B_SIZE" --edge_method "$EDGE_M" --edge_threshold "$EDGE_T" \
    --aggregation_graph "$AGG_G" --max_grad_norm "$MAX_NORM" --accumulation_steps "$ACC_STEPS" --dropout "$DROPOUT" \
    --seed "$SEED" --n_heads "$N_HEADS" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" --epoch_start "$START" \
    --wandb_mode "$WB_MODE";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Skipping TRANSGCN optimization (TRAIN_TRANSGCN=$TRAIN_TRANSGCN)"
fi

if [ "$TRAIN_DDAM" -eq 0 ]; then
    echo ""
    echo -e "${BLUE}===== [OPTIM] DDAM model =====${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py hp_optim --problem "$PROBLEM" --model ddam --encoder gat --hop_epochs "$H_EPOCHS" --waste_type "$WTYPE" \
    --data_distribution "$DATA_DIST" --n_epochs "$EPOCHS" --vertex_method "$VERTEX_M" --eta "$ETA" --mutpb "$MUTPB" \
    --batch_size "$B_SIZE" --epoch_size "$N_DATA" --val_size "$N_VAL_DATA" --n_pop "$N_POP" --temporal_horizon "${HORIZON[3]}" \
    --train_time --train_dataset "$DATASET" --normalization "$NORM"  --embedding_dim "$EMBED_DIM" --cxpb "$CXPB" \
    --activation "$ACTIVATION" --n_encode_layers "$N_ENC_L" --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --exp_beta "$EXP_BETA" \
    --graph_size "$SIZE" --no_tensorboard --range "${RANGE[@]}" --fevals "$FEVALS" --metric "$METRIC" --area "$AREA" \
    --focus_size "$F_SIZE" --eval_focus_size "$VAL_F_SIZE" --focus_graph "$FOCUS_GRAPH" --n_decode_layers "$N_DEC_L" \
    --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" --lr_model "$LR_MODEL" --lr_critic_value "$LR_CV" --dropout "$DROPOUT" \
    --eval_batch_size "$VAL_B_SIZE" --aggregation_graph "$AGG_G" --max_grad_norm "$MAX_NORM" --accumulation_steps "$ACC_STEPS" \
    --seed "$SEED" --n_heads "$N_HEADS" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" --epoch_start "$START" \
    --wandb_mode "$WB_MODE";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Skipping DDAM optimization (TRAIN_DDAM=$TRAIN_DDAM)"
fi

if [ "$TRAIN_TAM" -eq 0 ]; then
    echo ""
    echo -e "${BLUE}===== [OPTIM] TAM model =====${NC}"
    if [ "$VERBOSE" = false ]; then
        exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
        exec 3>&- 4>&-  # Close the temporary file descriptors
    fi
    python main.py hp_optim --problem "$PROBLEM" --model tam --encoder gat --hop_epochs "$H_EPOCHS" --waste_type "$WTYPE" \
    --metric "$METRIC" --data_distribution "$DATA_DIST" --n_epochs "$EPOCHS" --vertex_method "$VERTEX_M" --mutpb "$MUTPB" \
    --batch_size "$B_SIZE" --epoch_size "$N_DATA" --val_size "$N_VAL_DATA" --n_pop "$N_POP" --temporal_horizon "$HORIZON" \
    --area "$AREA" --train_time --train_dataset "$DATASET" --normalization "$NORM"  --embedding_dim "$EMBED_DIM" --cxpb "$CXPB" \
    --activation "$ACTIVATION" --n_encode_layers "$N_ENC_L" --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --exp_beta "$EXP_BETA" \
    --graph_size "$SIZE" --no_tensorboard --range "${RANGE[@]}" --fevals "$FEVALS" --temporal_horizon "${HORIZON[4]}" \
    --focus_size "$F_SIZE" --eval_focus_size "$VAL_F_SIZE" --focus_graph "$FOCUS_GRAPH" --n_predict_layers "$N_PRED_L" --eta "$ETA" \
    --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" --lr_model "$LR_MODEL" --lr_critic_value "$LR_CV" --dropout "$DROPOUT" \
    --eval_batch_size "$VAL_B_SIZE" --aggregation_graph "$AGG_G" --max_grad_norm "$MAX_NORM" --accumulation_steps "$ACC_STEPS" \
    --seed "$SEED" --n_heads "$N_HEADS" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" --epoch_start "$START" \
    --wandb_mode "$WB_MODE";
    if [ "$VERBOSE" = false ]; then
        exec >/dev/null 2>&1
    fi
else
    echo -e "${YELLOW}[SKIP]${NC} Skipping TAM optimization (TRAIN_TAM=$TRAIN_TAM)"
fi

echo ""
echo -e "${GREEN}✓ [DONE] Hyperparameter optimization completed.${NC}"