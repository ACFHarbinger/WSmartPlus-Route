from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class CPSATConfig:
    """
    Configuration for the Constraint Programming (CP-SAT) policy.

    Attributes:
        num_days: Multi-day horizon.
        time_limit: Maximum solve time in seconds.
        search_workers: Number of parallel workers for CP-SAT.
        mip_gap: Relative gap for optimality (conceptually similar in CP-SAT).
        scaling_factor: Precision factor for integer conversion (default 1000).
        waste_weight: Weight for waste collected in the objective.
        cost_weight: Weight for travel cost in the objective.
        overflow_penalty: Penalty per unit of overflow.
        mean_increment: Expected daily waste increment for bins.
        seed: Random seed for reproducibility.
        must_go: Node selection strategy configuration.
        post_processing: Post-processing operations.
    """

    num_days: int = 3
    time_limit: float = 300.0
    search_workers: int = 8
    mip_gap: float = 0.01
    scaling_factor: int = 1000

    waste_weight: float = 1.0
    cost_weight: float = 1.0
    overflow_penalty: float = 10.0
    mean_increment: float = 0.2

    seed: Optional[int] = None

    must_go: Optional[List[MustGoConfig]] = None
    post_processing: Optional[List[PostProcessingConfig]] = None
