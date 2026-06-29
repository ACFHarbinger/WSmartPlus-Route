"""
Infrastructure setup utilities module.

Attributes:
    setup_env: Sets up the environment for the simulation.
    setup_manager: Initializes and loads the Manager model for Hierarchical RL.
    setup_sims: Setup utilities to run simulations.
    setup_worker: Sets up and loads a specific model based on policy.

Example:
    setup_env(sim_cfg, device)
    setup_hrl_manager(sim_cfg, device)
    deep_sanitize(sim_cfg)
    get_pol_name(sim_cfg)
    get_graph_config(sim_cfg)
    setup_model("policy_name", "path/to/models", {"policy_name": "model.pt"}, device, lock)
"""

from .setup_env import setup_env
from .setup_manager import setup_hrl_manager
from .setup_sims import deep_sanitize, get_pol_name, get_graph_config
from .setup_worker import setup_model

__all__ = [
    "setup_env",
    "setup_hrl_manager",
    "deep_sanitize",
    "get_pol_name",
    "get_graph_config",
    "setup_model",
]
