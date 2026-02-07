"""
Training common subpackage for WSmart-Route.

This package contains training utilities and common components
like epoch preparation, dataset regeneration, and training hooks.
"""

from logic.src.pipeline.rl.common.base import RL4COLitModule
from logic.src.pipeline.rl.common.baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    MeanBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    SharedBaseline,
    WarmupBaseline,
    get_baseline,
)
from logic.src.pipeline.rl.common.reward_scaler import BatchRewardScaler, RewardScaler
from logic.src.pipeline.rl.common.trainer import WSTrainer

__all__ = [
    "RL4COLitModule",
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "WarmupBaseline",
    "POMOBaseline",
    "MeanBaseline",
    "SharedBaseline",
    "get_baseline",
    "BASELINE_REGISTRY",
    "WSTrainer",
    "RewardScaler",
    "BatchRewardScaler",
]
