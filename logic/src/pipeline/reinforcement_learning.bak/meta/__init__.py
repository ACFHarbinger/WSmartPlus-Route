"""
Meta-Learning Module for Adaptive Weight Optimization.

This module provides meta-learning capabilities for dynamic adjustment of reward/cost weights
during training, including:
- Reward Weight Adaptation (RWA)
- Contextual Bandits
- Temporal Difference Learning (TDL)
- Multi-Objective RL (MORL)
- HyperNetwork Optimization
"""

from .contextual_bandits import WeightContextualBandit
from .meta_trainers import (
    ContextualBanditTrainer,
    HRLTrainer,
    HyperNetworkTrainer,
    MORLTrainer,
    RWATrainer,
    TDLTrainer,
)
from .multi_objective import MORLWeightOptimizer
from .temporal_difference_learning import CostWeightManager
from .weight_optimizer import RewardWeightOptimizer
from .weight_strategy import WeightAdjustmentStrategy

__all__ = [
    "WeightAdjustmentStrategy",
    "CostWeightManager",
    "MORLWeightOptimizer",
    "RewardWeightOptimizer",
    "WeightContextualBandit",
    "ContextualBanditTrainer",
    "HRLTrainer",
    "HyperNetworkTrainer",
    "MORLTrainer",
    "RWATrainer",
    "TDLTrainer",
]
