"""
Configuration utilities module.

Attributes:
    setup_env: Sets up the environment for the simulation.
    setup_hrl_manager: Initializes and loads the Manager model for Hierarchical RL.
    setup_model: Sets up and loads a specific model based on policy.

Example:
    setup_env(sim_cfg, device)
    setup_hrl_manager(sim_cfg, device)
    setup_model("policy_name", "path/to/models", {"policy_name": "model.pt"}, device, lock)
"""

from .setup_env import setup_env
from .setup_manager import setup_hrl_manager
from .setup_worker import setup_model

__all__ = [
    "setup_env",
    "setup_hrl_manager",
    "setup_model",
]
