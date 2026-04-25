"""Checkpointing system for long-running simulations.

Attributes:
    None

Example:
    None
"""

from pathlib import Path

from .hooks import CheckpointHook
from .manager import CheckpointError, checkpoint_manager
from .persistence import SimulationCheckpoint

ROOT_DIR = Path(__file__).resolve().parent

__all__ = [
    "SimulationCheckpoint",
    "CheckpointHook",
    "CheckpointError",
    "checkpoint_manager",
    "ROOT_DIR",
]
