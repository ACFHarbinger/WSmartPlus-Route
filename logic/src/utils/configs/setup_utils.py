"""
Environment, model, and optimizer setup utilities.

This file acts as a facade for the setup sub-package.
"""

from .setup import (
    setup_cost_weights,
    setup_env,
    setup_hrl_manager,
    setup_model,
    setup_model_and_baseline,
    setup_optimizer_and_lr_scheduler,
)

__all__ = [
    "setup_cost_weights",
    "setup_env",
    "setup_hrl_manager",
    "setup_model",
    "setup_model_and_baseline",
    "setup_optimizer_and_lr_scheduler",
]
