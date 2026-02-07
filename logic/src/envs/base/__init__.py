"""
Base environment definitions.
"""

from .improvement import ImprovementEnvBase
from .rl4co import RL4COEnvBase

__all__ = ["RL4COEnvBase", "ImprovementEnvBase"]
