"""
Env Config module.

Attributes:
    EnvConfig: Environment configuration.

Example:
    >>> from logic.src.configs.envs import EnvConfig
    >>> config = EnvConfig()
    >>> print(config)
    EnvConfig(name='vrpp', min_loc=0.0, max_loc=1.0, capacity=None, data_distribution=None, min_fill=0.0, max_fill=1.0, fill_distribution='uniform', stochastic=False, mean=0.0, variance=0.0, temporal_horizon=0, curriculum_graphs=[], eval_graphs=[])
"""

from dataclasses import dataclass, field
from typing import List, Optional

from .graph import GraphConfig


@dataclass
class EnvConfig:
    """Environment configuration.

    All graph and reward settings are specified per-graph inside
    ``curriculum_graphs`` and ``eval_graphs``.  The first entry in
    ``curriculum_graphs`` acts as the primary training graph.

    Attributes:
        name: Name of the environment (e.g., 'vrpp', 'wcvrp').
        min_loc: Minimum coordinate value.
        max_loc: Maximum coordinate value.
        capacity: Vehicle capacity (optional).
        curriculum_graphs: Ordered list of graphs for sequential curriculum
            learning. The **first entry** is used for single-stage training.
            Each :class:`GraphConfig` entry carries an optional ``reward`` field
            to override objective weights for that stage.
        eval_graphs: List of graphs used for validation. Each entry may also
            carry an optional ``reward`` field.
    """

    name: str = "vrpp"
    min_loc: float = 0.0
    max_loc: float = 1.0
    capacity: Optional[float] = None
    # Data distribution and generation
    data_distribution: Optional[str] = None
    min_fill: float = 0.0
    max_fill: float = 1.0
    fill_distribution: str = "uniform"
    stochastic: bool = False
    mean: float = 0.0
    variance: float = 0.0
    temporal_horizon: int = 0
    curriculum_graphs: List[GraphConfig] = field(default_factory=list)
    eval_graphs: List[GraphConfig] = field(default_factory=list)
