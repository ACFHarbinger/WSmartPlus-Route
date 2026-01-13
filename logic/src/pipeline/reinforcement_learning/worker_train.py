"""
Worker Training Script.

This module contains the training loops for the "Worker" agent (the routing model),
implementing various Reinforcement Learning algorithms and strategies.

It serves as a dispatcher and collection of training routines for:
- Standard REINFORCE baseline.
- Meta-Learning extensions (Contextual Bandits, TDL, MORL).
- Hypernetwork-based training.
- Hierarchical RL (HRL) integration.
- Specialized algorithms (PPO, SAPO, GSPO, DR-GRPO) via their respective Trainer classes.

Key Functions:
- `train`: The main entry point that dispatches to specific training functions based on options.
- `train_reinforce_over_time`: Standard REINFORCE training loop.
"""
from logic.src.pipeline.reinforcement_learning.core.reinforce import (
    StandardTrainer, RWATrainer, ContextualBanditTrainer, 
    TDLTrainer, MORLTrainer, HyperNetworkTrainer, HRLTrainer, 
    TimeTrainer
)
from logic.src.pipeline.reinforcement_learning.core.ppo import PPOTrainer
from logic.src.pipeline.reinforcement_learning.core.sapo import SAPOTrainer
from logic.src.pipeline.reinforcement_learning.core.sapo import SAPOTrainer
from logic.src.pipeline.reinforcement_learning.core.gspo import GSPOTrainer
from logic.src.pipeline.reinforcement_learning.core.dr_grpo import DRGRPOTrainer


def train_reinforce_epoch(model, optimizer, baseline, lr_scheduler, scaler, epoch, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Wrapper for running a single epoch of standard training.

    Executes a training epoch using the appropriate trainer based on the configuration.
    Handles legacy logic where the epoch loop was external.
    """
    if opts.get('rl_algorithm') == 'ppo':
        trainer = PPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('rl_algorithm') == 'sapo':
        trainer = SAPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('rl_algorithm') == 'gspo':
        trainer = GSPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('rl_algorithm') == 'dr_grpo':
        trainer = DRGRPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    else:
        trainer = StandardTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    
    trainer.day = epoch # Set current epoch context
    trainer.initialize_training_dataset() # Initialize dataset for this epoch
    trainer.update_context() # Prepare dataset
    trainer.train_day() # Run training
    trainer.post_day_processing()
    return model, None


def train_reinforce_over_time_rwa(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    REINFORCE training loop with Relative Weight Adaptation (RWA).

    Dynamically adjusts objective weights by comparing the relative performance of different
    reward components across epochs.
    """
    trainer = RWATrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_reinforce_over_time_cb(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    REINFORCE training loop with Meta-Learning via Contextual Bandits.

    Adapts cost/reward weights dynamically using a Contextual Bandit strategy (e.g., UCB, Thompson Sampling).
    """
    trainer = ContextualBanditTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_reinforce_over_time_tdl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    REINFORCE training loop with Temporal Difference Learning (TDL) for weight adjustment.

    Uses TD-errors to adjust cost weights over time.
    """
    trainer = TDLTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_reinforce_over_time_morl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    REINFORCE training loop with Multi-Objective RL (MORL).

    Uses Pareto-front exploration strategies to balance multiple objectives during training.
    """
    trainer = MORLTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.weight_optimizer


def train_over_time_with_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Training loop using a Hypernetwork Meta-Learner.

    Trains a hypernetwork to generate weights for the main policy network.
    """
    trainer = HyperNetworkTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.hyper_optimizer


def train_reinforce_over_time_hrl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Hierarchical Reinforcement Learning (HRL) training loop.

    Trains the worker agent within a hierarchical framework, potentially interacting with a Manager agent.
    """
    trainer = HRLTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    trainer.train()
    return model, trainer.hrl_manager


def train_reinforce_over_time(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts):
    """
    Standard REINFORCE training loop implementation.
    Dispatches to specific variants (RWA, Contextual Bandit, TDL, MORL, HRL) or runs the default loop.
    Note: For 'standard', 'ppo', etc., this function delegates to specific loop implementations
    often found in the `train.py` script or legacy handlers.
    (This function seems to be the one referred to as `train` in `train.py` imports?)
    """
    if opts.get('rwa', False):
        return train_reinforce_over_time_rwa(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('cb', False):
        return train_reinforce_over_time_cb(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('tdl', False):
        return train_reinforce_over_time_tdl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('morl', False):
        return train_reinforce_over_time_morl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('hypernetwork', False):
        return train_over_time_with_hypernetwork(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('hrl', False):
        return train_reinforce_over_time_hrl(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
    elif opts.get('rl_algorithm') == 'ppo':
        trainer = PPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        trainer.train()
        return model, None
    elif opts.get('rl_algorithm') == 'sapo':
        trainer = SAPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        trainer.train()
        return model, None
    elif opts.get('rl_algorithm') == 'gspo':
        trainer = GSPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        trainer.train()
        return model, None
    elif opts.get('rl_algorithm') == 'dr_grpo':
        trainer = DRGRPOTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        trainer.train()
        return model, None
    else:
        # Default simple time training (Standard Time ?)
        # Use TimeTrainer
        trainer = TimeTrainer(model, optimizer, baseline, lr_scheduler, scaler, val_dataset, problem, tb_logger, cost_weights, opts)
        trainer.train()
        return model, None
