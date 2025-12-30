#!/bin/bash

# Handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Default to quiet mode
VERBOSE=false

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

EDGE_T=1.0
EDGE_M="knn"
DIST_M="gmaps"
VERTEX_M="mmn"

W_LEN=10.0
W_OVER=10.0
W_WASTE=10.0
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
LR_MODEL=0.0001
LR_SCHEDULER="lambda"
LR_DECAY=1.0

LOG_STEP=10
B_SIZE=128
N_DATA=1280
N_VAL_DATA=0 #1280
VAL_B_SIZE=0

BL="exponential"
POMO_SIZE=0
MAX_NORM=1.0
EXP_BETA=0.8
BL_ALPHA=0.05
ACC_STEPS=1

IMITATION_W=0.1
IMITATION_DECAY=0.91
IMITATION_DECAY_STEP=1
TWO_OPT_MAX_ITER=100

SIZE=100
AREA="riomaior"
WTYPE="plastic"
F_SIZE=1280
VAL_F_SIZE=0
DM_METHOD="gmaps"
F_GRAPH="graphs_${SIZE}V_1N_${WTYPE}.json"
DM_PATH="data/wsr_simulator/distance_matrix/gmaps_distmat_${WTYPE}[${AREA}].csv"
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

echo "Starting training script..."
echo "Problem: $PROBLEM"
echo "Graph size: $SIZE"
echo "Area: $AREA"
echo "Epochs: $EPOCHS"
echo ""

for ((id = 0; id < ${#DATA_DISTS[@]}; id++)); do
    echo "Processing data distribution: ${DATA_DISTS[id]}"
    if [ "$TRAIN_AM" -eq 0 ]; then
        echo "===== Training AM model ====="
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py train --problem "$PROBLEM" --model am --encoder gat --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --train_dataset "${DATASETS[id]}" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" \
        --temporal_horizon "${HORIZON[0]}" --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY" \
        --batch_size "$B_SIZE" --pomo_size "$POMO_SIZE" --bl_alpha "$BL_ALPHA" --area "$AREA" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --log_step "$LOG_STEP" --spatial_bias \
        --imitation_weight "$IMITATION_W" --imitation_decay "$IMITATION_DECAY" \
        --imitation_decay_step "$IMITATION_DECAY_STEP" --two_opt_max_iter "$TWO_OPT_MAX_ITER";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo "Skipping AM training (TRAIN_AM=$TRAIN_AM)"
    fi
    if [ "$TRAIN_AMGC" -eq 0 ]; then
        echo ""
        echo "===== Training AMGC model ====="
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py train --problem "$PROBLEM" --model am --encoder ggac --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --train_dataset "${DATASETS[id]}" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" \
        --temporal_horizon "${HORIZON[1]}" --lr_scheduler "$LR_SCHEDULER" --lr_decay "$LR_DECAY"  \
        --batch_size "$B_SIZE" --pomo_size "$POMO_SIZE" --bl_alpha "$BL_ALPHA" --area "$AREA" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --log_step "$LOG_STEP" --spatial_bias \
        --imitation_weight "$IMITATION_W" --imitation_decay "$IMITATION_DECAY" \
        --imitation_decay_step "$IMITATION_DECAY_STEP" --two_opt_max_iter "$TWO_OPT_MAX_ITER";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo "Skipping AMGC training (TRAIN_AMGC=$TRAIN_AMGC)"
    fi

    if [ "$TRAIN_TRANSGCN" -eq 0 ]; then
        echo ""
        echo "===== Training TRANSGCN model ====="
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py train --problem "$PROBLEM" --model am --encoder tgc --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --train_dataset "${DATASETS[id]}" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" --area "$AREA" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" --seed "$SEED" \
        --temporal_horizon "${HORIZON[2]}" --lr_scheduler "$LR_SCHEDULER" --n_encode_sublayers "$N_ENC_SL" \
        --batch_size "$B_SIZE" --pomo_size "$POMO_SIZE" --bl_alpha "$BL_ALPHA" --lr_decay "$LR_DECAY" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --log_step "$LOG_STEP" --spatial_bias \
        --imitation_weight "$IMITATION_W" --imitation_decay "$IMITATION_DECAY" \
        --imitation_decay_step "$IMITATION_DECAY_STEP" --two_opt_max_iter "$TWO_OPT_MAX_ITER";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo "Skipping TRANSGCN training (TRAIN_TRANSGCN=$TRAIN_TRANSGCN)"
    fi

    if [ "$TRAIN_DDAM" -eq 0 ]; then
        echo ""
        echo "===== Training DDAM model ====="
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py train --problem "$PROBLEM" --model ddam --encoder gat --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --train_dataset "${DATASETS[id]}" --area "$AREA" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" \
        --temporal_horizon "${HORIZON[3]}" --lr_scheduler "$LR_SCHEDULER" --n_decode_layers "$N_DEC_L"  \
        --batch_size "$B_SIZE" --pomo_size "$POMO_SIZE" --bl_alpha "$BL_ALPHA" --lr_decay "$LR_DECAY" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --log_step "$LOG_STEP" --spatial_bias \
        --imitation_weight "$IMITATION_W" --imitation_decay "$IMITATION_DECAY" \
        --imitation_decay_step "$IMITATION_DECAY_STEP" --two_opt_max_iter "$TWO_OPT_MAX_ITER";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo "Skipping DDAM training (TRAIN_DDAM=$TRAIN_DDAM)"
    fi

    if [ "$TRAIN_TAM" -eq 0 ]; then
        echo ""
        echo "===== Training TAM model ====="
        if [ "$VERBOSE" = false ]; then
            exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
            exec 3>&- 4>&-  # Close the temporary file descriptors
        fi
        python main.py train --problem "$PROBLEM" --model tam --encoder gat --epoch_size "$N_DATA" \
        --data_distribution "${DATA_DISTS[id]}" --graph_size "$SIZE" --n_epochs "$EPOCHS" --seed "$SEED" \
        --train_time --vertex_method "$VERTEX_M" --epoch_start "$START" --max_grad_norm "$MAX_NORM" \
        --val_size "$N_VAL_DATA" --w_length "$W_LEN" --w_waste "$W_WASTE" --w_overflows "$W_OVER" \
        --embedding_dim "$EMBED_DIM" --activation "$ACTI_F" --accumulation_steps "$ACC_STEPS" \
        --focus_graph "$F_GRAPH" --normalization "$NORM" --train_dataset "${DATASETS[id]}" --area "$AREA" \
        --optimizer "$OPTIM" --hidden_dim "$HIDDEN_DIM" --n_heads "$N_HEADS" --dropout "$DROPOUT" \
        --waste_type "$WTYPE" --focus_size "$F_SIZE" --n_encode_layers "$N_ENC_L" --lr_model "$LR_MODEL" \
        --eval_focus_size "$VAL_F_SIZE" --distance_method "$DIST_M" --exp_beta "$EXP_BETA" \
        --edge_threshold "$EDGE_T" --edge_method "$EDGE_M" --eval_batch_size "$VAL_B_SIZE" \
        --temporal_horizon "${HORIZON[4]}" --lr_scheduler "$LR_SCHEDULER" --n_predict_layers "$N_PRED_L"  \
        --batch_size "$B_SIZE" --pomo_size "$POMO_SIZE" --bl_alpha "$BL_ALPHA" --lr_decay "$LR_DECAY" \
        --aggregation_graph "$AGG_G" --distance_method "$DIST_METHOD" --dm_filepath "$DM_PATH" \
        --wandb_mode "$WB_MODE" --distance_method "$DM_METHOD" --log_step "$LOG_STEP" --spatial_bias \
        --imitation_weight "$IMITATION_W" --imitation_decay "$IMITATION_DECAY" \
        --imitation_decay_step "$IMITATION_DECAY_STEP" --two_opt_max_iter "$TWO_OPT_MAX_ITER";
        if [ "$VERBOSE" = false ]; then
            exec >/dev/null 2>&1
        fi
    else
        echo "Skipping TAM training (TRAIN_TAM=$TRAIN_TAM)"
    fi
done

echo ""
echo "===== Training completed for all distributions ====="