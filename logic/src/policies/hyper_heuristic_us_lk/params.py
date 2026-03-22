"""
HULK (Hyper-heuristic with Unstringing/stringing and LK) Parameters.

Reference:
    Müller, L. F., & Bonilha, I. (2022). "Hyper-Heuristic Based on ACO
    and Local Search for Dynamic Optimization Problems."
    Algorithms, 15(1), 9. https://doi.org/10.3390/a15010009
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class HULKParams:
    """Parameters for HULK hyper-heuristic."""

    # Search parameters
    max_iterations: int = 1000
    time_limit: float = 300.0
    restarts: int = 3
    restart_threshold: int = 100

    # Operator selection parameters
    epsilon: float = 0.3
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Performance tracking
    memory_size: int = 50

    # Acceptance criteria
    accept_worse_prob: float = 0.1
    acceptance_decay: float = 0.99

    # Temperature for simulated annealing
    start_temp: float = 100.0
    cooling_rate: float = 0.99
    min_temp: float = 0.01

    # Unstringing/Stringing parameters
    min_destroy_size: int = 2
    max_destroy_pct: float = 0.3

    # Local search parameters
    ls_intensity: int = 5
    local_search_iterations: int = 10
    local_search_operators: List[str] = field(
        default_factory=lambda: [
            "2-opt",
            "3-opt",
            "swap",
            "relocate",
        ]
    )

    # Unstringing operators (destroy)
    unstring_operators: List[str] = field(
        default_factory=lambda: [
            "type_i",
            "type_ii",
            "type_iii",
            "type_iv",
        ]
    )

    # Stringing operators (repair)
    string_operators: List[str] = field(
        default_factory=lambda: [
            "type_i",
            "type_ii",
            "type_iii",
            "type_iv",
        ]
    )

    # Müller & Bonilha (2022) Performance scoring (Alpha, Beta, Gamma, Delta)
    score_alpha: float = 20.0
    score_beta: float = 10.0
    score_gamma: float = 5.0
    score_delta: float = 0.5

    # Aliases
    score_best: float = 20.0
    score_improvement: float = 10.0
    score_accept: float = 5.0
    score_reject: float = 0.5

    # Operator weights
    operator_weights: dict = field(default_factory=dict)
    weight_learning_rate: float = 0.1
    weight_decay: float = 0.95

    def __post_init__(self):
        """Initialize operator weights if not provided."""
        if not self.operator_weights:
            all_ops = self.unstring_operators + self.string_operators + self.local_search_operators
            self.operator_weights = {op: 1.0 for op in all_ops}
