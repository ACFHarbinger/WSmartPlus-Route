"""
Hydra configuration for HULK hyper-heuristic.
"""

from dataclasses import dataclass, field
from typing import List, Optional

from hydra.core.config_store import ConfigStore


@dataclass
class HULKConfig:
    """Hydra config for HULK hyper-heuristic."""

    # Search parameters
    seed: Optional[int] = None
    vrpp: bool = True
    max_iterations: int = 1000
    time_limit: float = 300.0
    restarts: int = 3
    restart_threshold: int = 100

    # Operator selection
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Performance tracking
    memory_size: int = 50

    # Acceptance
    accept_worse_prob: float = 0.1
    acceptance_decay: float = 0.99

    # Simulated annealing
    start_temp: float = 100.0
    cooling_rate: float = 0.99
    min_temp: float = 0.01

    # Destruction/repair
    min_destroy_size: int = 2
    max_destroy_pct: float = 0.3

    # Local search
    local_search_iterations: int = 10
    local_search_operators: List[str] = field(default_factory=lambda: ["2-opt", "3-opt", "swap", "relocate"])

    # Unstringing operators
    unstring_operators: List[str] = field(default_factory=lambda: ["type_i", "type_ii", "type_iii", "type_iv"])

    # Stringing operators
    string_operators: List[str] = field(default_factory=lambda: ["type_i", "type_ii", "type_iii", "type_iv"])

    # Scoring
    score_improvement: float = 10.0
    score_accept: float = 5.0
    score_reject: float = -1.0
    score_best: float = 20.0

    # Weight learning
    weight_learning_rate: float = 0.1
    weight_decay: float = 0.95


def register_configs():
    """Register HULK configs with Hydra."""
    cs = ConfigStore.instance()
    cs.store(group="policies", name="hulk", node=HULKConfig)
