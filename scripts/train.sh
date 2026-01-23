#!/bin/bash

# Handle memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default to verbose mode
VERBOSE=true

# Handle --quiet if it appears after other arguments
for arg in "$@"; do
    if [[ "$arg" == "--quiet" ]]; then
        VERBOSE=false
    fi
done

# If not verbose, redirect all output to /dev/null
if [ "$VERBOSE" = false ]; then
    exec 3>&1 4>&2  # Save original stdout (fd1) to fd3, stderr (fd2) to fd4
    exec >/dev/null 2>&1
fi

# Hydra Config Overrides (replaces old CLI flags)
MODEL="am"
ENCODER="gat"
PROBLEM="wcvrp"
GRAPH_SIZE=100
AREA="riomaior"
SEED=42
EPOCHS=100

# Model architecture
EMBEDDING_DIM=128
HIDDEN_DIM=512
N_HEADS=8
N_ENCODE_LAYERS=3
N_DECODE_LAYERS=2
NORMALIZATION="instance"
ACTIVATION="gelu"
DROPOUT=0.1

# Training hyperparameters
RL_ALGO="reinforce"
OPTIMIZER="rmsprop"
LR_MODEL=0.0001
BATCH_SIZE=128
EPOCH_SIZE=128
MAX_GRAD_NORM=1.0

# Baseline settings
BASELINE="exponential"
BL_ALPHA=0.05
EXP_BETA=0.8

# Other settings
WANDB_MODE="disabled"  # 'online'|'offline'|'disabled'
LOG_STEP=10

# Data distribution
DATA_DIST="gamma1"


echo -e "${BLUE}Starting training script (Hydra-based)...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Model:      ${MAGENTA}$MODEL${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$GRAPH_SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Epochs:     ${MAGENTA}$EPOCHS${NC}"
echo -e "${CYAN}[CONFIG]${NC} Device:     ${MAGENTA}CUDA Accelerator${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

# Restore stdout/stderr if in verbose mode for command output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
    exec 3>&- 4>&-  # Close the temporary file descriptors
fi

echo -e "${BLUE}===== [TRAIN] $MODEL with $ENCODER encoder =====${NC}"

# Use train_lightning with Hydra config overrides
uv run python main.py train_lightning \
    model.name=$MODEL \
    model.encoder_type=$ENCODER \
    env.name=$PROBLEM \
    env.num_loc=$GRAPH_SIZE \
    env.data_distribution=$DATA_DIST \
    model.embed_dim=$EMBEDDING_DIM \
    model.hidden_dim=$HIDDEN_DIM \
    model.num_heads=$N_HEADS \
    model.num_encoder_layers=$N_ENCODE_LAYERS \
    model.normalization=$NORMALIZATION \
    model.activation=$ACTIVATION \
    model.dropout=$DROPOUT \
    optim.optimizer=$OPTIMIZER \
    optim.lr=$LR_MODEL \
    train.batch_size=$BATCH_SIZE \
    train.train_data_size=$EPOCH_SIZE \
    train.n_epochs=$EPOCHS \
    rl.max_grad_norm=$MAX_GRAD_NORM \
    rl.baseline=$BASELINE \
    rl.bl_alpha=$BL_ALPHA \
    rl.exp_beta=$EXP_BETA \
    seed=$SEED \
    train.log_step=$LOG_STEP \
    wandb_mode=$WANDB_MODE \
    "$@"  # Pass additional arguments from command line

# Redirect to /dev/null again if quiet
if [ "$VERBOSE" = false ]; then
    exec >/dev/null 2>&1
fi

echo ""
echo -e "${GREEN}✓ [DONE] Training completed${NC}"
