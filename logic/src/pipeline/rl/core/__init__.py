"""
RL Pipeline module for WSmart-Route.
"""
from logic.src.pipeline.rl.core.adaptive_imitation import AdaptiveImitation
from logic.src.pipeline.rl.core.base import RL4COLitModule
from logic.src.pipeline.rl.core.baselines import (
    BASELINE_REGISTRY,
    Baseline,
    CriticBaseline,
    ExponentialBaseline,
    NoBaseline,
    POMOBaseline,
    RolloutBaseline,
    WarmupBaseline,
    get_baseline,
)
from logic.src.pipeline.rl.core.dr_grpo import DRGRPO
from logic.src.pipeline.rl.core.gdpo import GDPO
from logic.src.pipeline.rl.core.gspo import GSPO
from logic.src.pipeline.rl.core.hrl import HRLModule
from logic.src.pipeline.rl.core.imitation import ImitationLearning
from logic.src.pipeline.rl.core.pomo import POMO
from logic.src.pipeline.rl.core.ppo import PPO
from logic.src.pipeline.rl.core.reinforce import REINFORCE
from logic.src.pipeline.rl.core.sapo import SAPO
from logic.src.pipeline.rl.core.symnco import SymNCO
from logic.src.pipeline.rl.meta.module import MetaRLModule

__all__ = [
    "RL4COLitModule",
    "Baseline",
    "NoBaseline",
    "ExponentialBaseline",
    "RolloutBaseline",
    "CriticBaseline",
    "WarmupBaseline",
    "POMOBaseline",
    "get_baseline",
    "BASELINE_REGISTRY",
    "REINFORCE",
    "PPO",
    "SAPO",
    "GSPO",
    "GDPO",
    "AdaptiveImitation",
    "DRGRPO",
    "ImitationLearning",
    "MetaRLModule",
    "HRLModule",
    "POMO",
    "SymNCO",
]
