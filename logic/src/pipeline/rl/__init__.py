"""
RL Pipeline module for WSmart-Route.
"""
from logic.src.pipeline.rl.base import RL4COLitModule
from logic.src.pipeline.rl.baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    NoBaseline,
    RolloutBaseline,
    get_baseline,
)
from logic.src.pipeline.rl.ppo import PPO
from logic.src.pipeline.rl.reinforce import REINFORCE

__all__ = [
    "RL4COLitModule",
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "get_baseline",
    "BASELINE_REGISTRY",
    "REINFORCE",
    "PPO",
]
