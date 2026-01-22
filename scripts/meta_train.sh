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

# Hydra Config Overrides for Meta-RL Training
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
NORMALIZATION="instance"
ACTIVATION="gelu"
DROPOUT=0.1

# Training hyperparameters
OPTIMIZER="rmsprop"
LR_MODEL=0.00005
LR_CRITIC=0.0001
BATCH_SIZE=256
EPOCH_SIZE=1280
MAX_GRAD_NORM=1.0

# Baseline settings
BASELINE="exponential"
BL_ALPHA=0.05
EXP_BETA=0.8

# Meta-RL / HRL settings
META_METHOD="hrl"  # Options: hrl, rnn, bandit, morl, tdl, hypernet
HRL_EPOCHS=2
HRL_CLIP_EPS=0.2
HRL_THRESHOLD=0.9
HRL_PID_TARGET=0.0003
HRL_LAMBDA_WASTE=300.0
HRL_LAMBDA_COST=0.5
HRL_LAMBDA_OVERFLOW_INIT=2000.0
HRL_LAMBDA_OVERFLOW_MIN=100.0
HRL_LAMBDA_OVERFLOW_MAX=5000.0
HRL_ENTROPY=0.01

# Manager architecture (for HRL)
GAT_HIDDEN=128
LSTM_HIDDEN=64
GATE_THRESH=0.5

# Other settings
WANDB_MODE="disabled"  # 'online'|'offline'|'disabled'
LOG_STEP=10
DATA_DIST="gamma1"

echo -e "${BLUE}Starting Meta-RL Training Module (Hydra-based)...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Model:      ${MAGENTA}$MODEL${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$GRAPH_SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Epochs:     ${MAGENTA}$EPOCHS${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}$META_METHOD${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

# Restore stdout/stderr if in verbose mode for command output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
    exec 3>&- 4>&-  # Close the temporary file descriptors
fi

echo -e "${BLUE}===== [TRAIN] $MODEL with $META_METHOD =====${NC}"

# Use train_lightning with Meta-RL experiment config
uv run python main.py train_lightning \
    experiment=meta_rl \
    model=$MODEL \
    model.encoder_type=$ENCODER \
    env.name=$PROBLEM \
    env.num_loc=$GRAPH_SIZE \
    env.generator.data_dist=$DATA_DIST \
    model.embedding_dim=$EMBEDDING_DIM \
    model.hidden_dim=$HIDDEN_DIM \
    model.num_heads=$N_HEADS \
    model.num_layers=$N_ENCODE_LAYERS \
    model.normalization=$NORMALIZATION \
    model.activation=$ACTIVATION \
    model.dropout=$DROPOUT \
    train.optimizer=$OPTIMIZER \
    train.lr=$LR_MODEL \
    train.batch_size=$BATCH_SIZE \
    train.epoch_size=$EPOCH_SIZE \
    train.n_epochs=$EPOCHS \
    train.max_grad_norm=$MAX_GRAD_NORM \
    train.baseline=$BASELINE \
    train.bl_alpha=$BL_ALPHA \
    train.exp_beta=$EXP_BETA \
    train.seed=$SEED \
    train.log_step=$LOG_STEP \
    logger.wandb.mode=$WANDB_MODE \
    meta_rl.method=$META_METHOD \
    meta_rl.manager.gat_hidden=$GAT_HIDDEN \
    meta_rl.manager.lstm_hidden=$LSTM_HIDDEN \
    meta_rl.manager.gate_threshold=$GATE_THRESH \
    meta_rl.hrl.epochs=$HRL_EPOCHS \
    meta_rl.hrl.clip_eps=$HRL_CLIP_EPS \
    meta_rl.hrl.threshold=$HRL_THRESHOLD \
    meta_rl.hrl.pid_target=$HRL_PID_TARGET \
    meta_rl.hrl.lambda_waste=$HRL_LAMBDA_WASTE \
    meta_rl.hrl.lambda_cost=$HRL_LAMBDA_COST \
    meta_rl.hrl.lambda_overflow_initial=$HRL_LAMBDA_OVERFLOW_INIT \
    meta_rl.hrl.lambda_overflow_min=$HRL_LAMBDA_OVERFLOW_MIN \
    meta_rl.hrl.lambda_overflow_max=$HRL_LAMBDA_OVERFLOW_MAX \
    meta_rl.hrl.entropy_coef=$HRL_ENTROPY \
    train.lr_critic=$LR_CRITIC \
    "$@"  # Pass additional arguments from command line

# Redirect to /dev/null again if quiet
if [ "$VERBOSE" = false ]; then
    exec >/dev/null 2>&1
fi

echo ""
echo -e "${GREEN}✓ [DONE] Meta-RL Training completed${NC}"
