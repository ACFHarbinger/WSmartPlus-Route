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
from logic.src.pipeline.rl.dr_grpo import DRGRPO
from logic.src.pipeline.rl.gspo import GSPO
from logic.src.pipeline.rl.hrl import HRLModule
from logic.src.pipeline.rl.meta import MetaRLModule
from logic.src.pipeline.rl.ppo import PPO
from logic.src.pipeline.rl.reinforce import REINFORCE
from logic.src.pipeline.rl.sapo import SAPO

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
    "SAPO",
    "GSPO",
    "DRGRPO",
    "MetaRLModule",
    "HRLModule",
]
