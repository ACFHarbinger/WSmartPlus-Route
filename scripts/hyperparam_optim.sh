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

# Hydra Config Overrides for HPO
MODEL="am"
ENCODER="gat"
PROBLEM="wcvrp"
GRAPH_SIZE=20
AREA="Rio Maior"
SEED=42
EPOCHS=7  # Budget per trial

# Model architecture defaults
EMBEDDING_DIM=128
HIDDEN_DIM=512
N_HEADS=8
N_ENCODE_LAYERS=3
NORMALIZATION="instance"
ACTIVATION="gelu"
DROPOUT=0.1

# Training hyperparameters defaults
OPTIMIZER="rmsprop"
LR_MODEL=0.0001
BATCH_SIZE=256
EPOCH_SIZE=128000
MAX_GRAD_NORM=1.0

# Baseline settings
BASELINE="exponential"
BL_ALPHA=0.05
EXP_BETA=0.8

# HPO settings
HPO_METHOD="dehb"  # Options: dehb, optuna_tpe, optuna_grid, optuna_random, optuna_hyperband
HPO_N_TRIALS=25
HPO_MIN_BUDGET=1
HPO_MAX_BUDGET=7
HPO_N_WORKERS=4

# Optuna-specific (if using optuna methods)
OPTUNA_SAMPLER="tpe"  # tpe, random, grid

# Other settings
WANDB_MODE="disabled"  # 'online'|'offline'|'disabled'
DATA_DIST="gamma1"

echo -e "${BLUE}Starting Hyperparameter Optimization (Hydra-based)...${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo -e "${CYAN}[CONFIG]${NC} Problem:    ${MAGENTA}$PROBLEM${NC}"
echo -e "${CYAN}[CONFIG]${NC} Model:      ${MAGENTA}$MODEL${NC}"
echo -e "${CYAN}[CONFIG]${NC} Graph size: ${MAGENTA}$GRAPH_SIZE${NC}"
echo -e "${CYAN}[CONFIG]${NC} Area:       ${MAGENTA}$AREA${NC}"
echo -e "${CYAN}[CONFIG]${NC} Method:     ${MAGENTA}$HPO_METHOD${NC}"
echo -e "${CYAN}[CONFIG]${NC} Trials:     ${MAGENTA}$HPO_N_TRIALS${NC}"
echo -e "${CYAN}---------------------------------------${NC}"
echo ""

# Restore stdout/stderr if in verbose mode for command output
if [ "$VERBOSE" = false ]; then
    exec 1>&3 2>&4  # Restore stdout from fd3, stderr from fd4
    exec 3>&- 4>&-  # Close the temporary file descriptors
fi

echo -e "${BLUE}===== [HPO] Optimizing $MODEL with $HPO_METHOD =====${NC}"

# Use train_lightning with HPO experiment config
uv run python main.py train_lightning \
    experiment=hpo \
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
    logger.wandb.mode=$WANDB_MODE \
    hpo.method=$HPO_METHOD \
    hpo.n_trials=$HPO_N_TRIALS \
    hpo.dehb.min_fidelity=$HPO_MIN_BUDGET \
    hpo.dehb.max_fidelity=$HPO_MAX_BUDGET \
    hpo.optuna.sampler=$OPTUNA_SAMPLER \
    hpo.n_workers=$HPO_N_WORKERS \
    "$@"  # Pass additional arguments from command line

# Redirect to /dev/null again if quiet
if [ "$VERBOSE" = false ]; then
    exec >/dev/null 2>&1
fi

echo ""
echo -e "${GREEN}✓ [DONE] Hyperparameter optimization completed${NC}"
