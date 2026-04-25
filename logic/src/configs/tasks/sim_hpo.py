"""
Simulation HPO Config module.

Attributes:
    SimHPOConfig: Configuration for simulation policy hyperparameter optimization.

Example:
    sim_hpo_config = SimHPOConfig(
        method="tpe",
        metric="profit",
        n_trials=20,
        num_workers=1,
        policy_name="alns",
        graph=GraphConfig(),
    )
"""

from dataclasses import dataclass, field
from typing import Any, Dict

from logic.src.configs.envs.graph import GraphConfig


@dataclass
class SimHPOConfig:
    """Configuration for simulation policy hyperparameter optimization.

    Attributes:
        method: HPO method ('tpe', 'random', 'grid', 'hyperband').
        metric: Optimization metric to maximize (e.g., 'profit').
        n_trials: Number of HPO trials.
        num_workers: Number of parallel workers for HPO.
        search_space: Dictionary defining the search space.
        policy_name: Name of the policy to optimize (e.g., 'alns').
        graph: Graph configuration for simulation trials.
    """

    method: str = "tpe"
    metric: str = "profit"
    n_trials: int = 20
    num_workers: int = 1
    search_space: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    policy_name: str = "alns"
    graph: GraphConfig = field(default_factory=GraphConfig)
