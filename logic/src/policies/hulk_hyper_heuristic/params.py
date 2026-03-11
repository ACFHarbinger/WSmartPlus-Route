"""
HULK (Hyper-heuristic Using unstringing/stringing with Local search and K-opt) Parameters.

Configuration for the HULK hyper-heuristic that combines:
- Unstringing operators for destruction
- Stringing operators for reconstruction
- Local search for improvement
- Adaptive operator selection based on performance

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
    time_limit: float = 300.0  # seconds
    restarts: int = 3
    restart_threshold: int = 100  # iterations without improvement

    # Operator selection parameters
    epsilon: float = 0.3  # exploration rate
    epsilon_decay: float = 0.995
    min_epsilon: float = 0.05

    # Performance tracking
    memory_size: int = 50  # window for operator performance

    # Acceptance criteria
    accept_worse_prob: float = 0.1
    acceptance_decay: float = 0.99

    # Temperature for simulated annealing
    start_temp: float = 100.0
    cooling_rate: float = 0.99
    min_temp: float = 0.01

    # Unstringing/Stringing parameters
    min_destroy_size: int = 2
    max_destroy_pct: float = 0.3  # max percentage of nodes to destroy

    # Local search parameters
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

    # Operator weights (initial, will be adapted)
    operator_weights: dict = field(default_factory=dict)

    # Performance scoring
    score_improvement: float = 10.0  # points for improvement
    score_accept: float = 5.0  # points for accepted move
    score_reject: float = -1.0  # penalty for rejection
    score_best: float = 20.0  # bonus for new best solution

    # Adaptive weight learning
    weight_learning_rate: float = 0.1
    weight_decay: float = 0.95  # decay old weights over time

    def __post_init__(self):
        """Initialize operator weights if not provided."""
        if not self.operator_weights:
            # Initialize all operators with equal weights
            all_ops = self.unstring_operators + self.string_operators + self.local_search_operators
            self.operator_weights = {op: 1.0 for op in all_ops}
