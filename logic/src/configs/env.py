"""
Env Config module.
"""

from dataclasses import dataclass
from typing import Optional


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
    overflow_penalty: float = 1.0
    collection_reward: float = 1.0
    cost_weight: float = 1.0
    prize_weight: float = 1.0
    # NEW FIELDS:
    area: str = "riomaior"
    waste_type: str = "plastic"
    focus_graph: Optional[str] = None
    focus_size: int = 0
    eval_focus_size: int = 0
    distance_method: str = "ogd"
    dm_filepath: Optional[str] = None
    waste_filepath: Optional[str] = None
    vertex_method: str = "mmn"
    edge_threshold: float = 0.0
    edge_method: Optional[str] = None
