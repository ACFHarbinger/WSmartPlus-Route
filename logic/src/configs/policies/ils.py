"""
ILS (Iterated Local Search) configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

from ..other.must_go import MustGoConfig
from ..other.post_processing import PostProcessingConfig


@dataclass
class ILSConfig:
    """Configuration for Iterated Local Search (ILS) policy.

    Attributes:
        n_restarts: Number of ILS restarts (perturbation cycles).
        ls_iterations: Iterations for local search within each phase.
        perturbation_strength: Fraction of tour to perturb.
        ls_operator: Local search operator name or dict of {name: prob}.
        perturbation_type: Perturbation method name or dict of {mode: prob}.
        time_limit: Maximum time in seconds for the solver.
        must_go: List of must-go strategy config files.
        post_processing: List of post-processing operations to apply.
    """

    n_restarts: int = 5
    ls_iterations: int = 50
    perturbation_strength: float = 0.2
    ls_operator: Union[str, Dict[str, float]] = "two_opt"
    perturbation_type: Union[str, Dict[str, float]] = "double_bridge"
    time_limit: float = 30.0
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
    perturb_probs: Dict[str, float] = field(
        default_factory=lambda: {
            "double_bridge": 0.5,
            "shuffle": 0.3,
            "random_swap": 0.2,
        }
    )
    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
