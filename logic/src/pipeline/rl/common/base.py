"""
PyTorch Lightning base module for RL training.

This file acts as a facade for the logic.src.pipeline.rl.common.base sub-package.
"""

from .base.model import RL4COLitModule

__all__ = ["RL4COLitModule"]
