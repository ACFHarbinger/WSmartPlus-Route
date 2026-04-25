"""
Env Config module.

Attributes:
    EnvConfig: Environment configuration.

Example:
    >>> from logic.src.configs.envs import EnvConfig
    >>> config = EnvConfig()
    >>> print(config)
    EnvConfig(name='vrpp', num_loc=50, min_loc=0.0, max_loc=1.0, capacity=None, graph=GraphConfig(num_loc=50, num_nodes=50, num_customers=20, customer_types={'A': 0.25, 'B': 0.5, 'C': 0.25}, area='alpine', demand_distribution='normal', demand_normal_mean=10, demand_normal_std=1, min_demand=1, max_demand=25, capacity_distribution='normal', capacity_normal_mean=300, capacity_normal_std=50, min_capacity=200, max_capacity=400, instance_generator='random', edge_probability=0.3, shuffle_seed=42), reward=ObjectiveConfig(w_capacity=1.0, w_distance=1.0, w_time=0.0, w_penalty=1.0, penalty_threshold=1.0, alpha_distance=1.0), data_distribution=None, min_fill=0.0, max_fill=1.0, fill_distribution='uniform', temporal_horizon=0)
"""

from dataclasses import dataclass, field
from typing import Optional

from .graph import GraphConfig
from .objective import ObjectiveConfig


@dataclass
class EnvConfig:
    """Environment configuration.

    Attributes:
        name: Name of the environment (e.g., 'vrpp', 'wcvrp').
        num_loc: Number of locations (excluding depot).
        min_loc: Minimum coordinate value.
        max_loc: Maximum coordinate value.
        capacity: Vehicle capacity (optional).
    """

    name: str = "vrpp"
    num_loc: int = 50
    min_loc: float = 0.0
    max_loc: float = 1.0
    capacity: Optional[float] = None
    graph: GraphConfig = field(default_factory=GraphConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    # Data distribution and generation
    data_distribution: Optional[str] = None
    min_fill: float = 0.0
    max_fill: float = 1.0
    fill_distribution: str = "uniform"
    temporal_horizon: int = 0
