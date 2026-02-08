"""
Post-processing configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class PostProcessingConfig:
    """Configuration for route refinement and post-processing strategies.

    Attributes:
        methods: List of post-processing methods to apply in sequence.
            Supported: 'fast_tsp', 'ils', '2opt', 'swap', 'relocate', '3opt', etc.
        iterations: Maximum number of local search iterations for refinement.
        n_restarts: Number of restarts/perturbations for Iterated Local Search (ILS).
        perturbation_strength: Strength of perturbation fraction for ILS escaping.
        ls_operator: Preferred local search operator (e.g., '2opt', 'swap_star').
        perturbation_type: Perturbation strategy (e.g., 'double_bridge', 'random').
        time_limit: Soft time limit for post-processing operations in seconds.
        params: Additional strategy-specific parameters as a dictionary.
    """

    methods: List[str] = field(default_factory=lambda: ["fast_tsp"])

    # Local Search & ILS parameters
    iterations: int = 50
    n_restarts: int = 5
    perturbation_strength: float = 0.2
    ls_operator: str = "2opt"
    perturbation_type: str = "double_bridge"

    # Execution parameters
    time_limit: float = 10.0

    # Additional parameters
    params: Dict[str, Union[int, float, str, bool]] = field(default_factory=dict)
