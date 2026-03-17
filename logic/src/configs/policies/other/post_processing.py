"""
Post-processing configuration module.

Defines structured configurations for route refinement and post-processing strategies,
mirroring the reinforcement learning configuration pattern.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class FastTSPPostConfig:
    """Configuration for Fast TSP optimization refinement.

    Attributes:
        time_limit: Maximum time in seconds for the solver.
        seed: Random seed for reproducibility.
    """

    time_limit: float = 2.0
    seed: int = 42


@dataclass
class LKHPostConfig:
    """Configuration for Lin-Kernighan-Helsgaun (LKH) refinement.

    Attributes:
        max_iterations: Maximum number of LKH iterations.
        seed: Random seed for reproducibility.
    """

    max_iterations: int = 1000
    seed: int = 42


@dataclass
class LocalSearchPostConfig:
    """Configuration for classical local search refinement.

    Attributes:
        operator_name: Local search operator (e.g., '2opt', 'swap', 'relocate').
        n_iterations: Maximum number of local search iterations.
        seed: Random seed for reproducibility.
    """

    operator_name: str = "2opt"
    n_iterations: int = 50
    seed: int = 42


@dataclass
class PathPostConfig:
    """Configuration for path-based refinement (opportunistic pickups).

    Attributes:
        vehicle_capacity: Maximum vehicle capacity for picking up extra nodes.
    """

    vehicle_capacity: float = 100.0


@dataclass
class RandomLocalSearchPostConfig:
    """Configuration for stochastic (random) local search refinement.

    Attributes:
        n_iterations: Number of random local search iterations.
        op_probs: Probabilities for selecting different local search operators.
        seed: Random seed for reproducibility.
    """

    n_iterations: int = 50
    op_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "two_opt": 0.25,
            "swap": 0.15,
            "relocate": 0.15,
            "two_opt_star": 0.2,
            "swap_star": 0.15,
            "three_opt": 0.1,
        }
    )
    seed: int = 42


@dataclass
class PostProcessingConfig:
    """Unified configuration for route refinement and post-processing strategies.

    Composes algorithm-specific parameters and execution settings into a single object.

    Attributes:
        methods: List of post-processing methods to apply in sequence.
            Supported: 'fast_tsp', 'lkh', 'classical_local_search', 'random_local_search', 'path'.
        fast_tsp: Configuration for Fast TSP solver.
        lkh: Configuration for Lin-Kernighan-Helsgaun solver.
        local_search: Configuration for classical local search.
        random_local_search: Configuration for random local search.
        path: Configuration for path-based refinement.
        time_limit: Soft global time limit for post-processing operations.
        params: Additional strategy-specific parameters as a dictionary.
    """

    methods: List[str] = field(default_factory=lambda: ["fast_tsp"])

    # Algorithm-specific sub-configs
    fast_tsp: FastTSPPostConfig = field(default_factory=FastTSPPostConfig)
    lkh: LKHPostConfig = field(default_factory=LKHPostConfig)
    local_search: LocalSearchPostConfig = field(default_factory=LocalSearchPostConfig)
    random_local_search: RandomLocalSearchPostConfig = field(default_factory=RandomLocalSearchPostConfig)
    path: PathPostConfig = field(default_factory=PathPostConfig)

    # Additional parameters
    params: Dict[str, Any] = field(default_factory=dict)
