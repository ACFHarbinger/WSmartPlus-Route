"""
Env Config module.
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
    min_loc: float = 0.0
    max_loc: float = 1.0
    graph: GraphConfig = field(default_factory=GraphConfig)
    reward: ObjectiveConfig = field(default_factory=ObjectiveConfig)
    # Data distribution and generation
    data_distribution: Optional[str] = None
    min_fill: float = 0.0
    max_fill: float = 1.0
    fill_distribution: str = "uniform"
