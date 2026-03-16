"""
Cluster-First Route-Second (CF-RS) configuration schemas.

This module defines the Hydra-compatible configuration dataclasses for the CF-RS
routing policy, following the project's standard configuration architecture.
"""

from dataclasses import dataclass
from typing import List, Optional

from .other.must_go import MustGoConfig
from .other.post_processing import PostProcessingConfig


@dataclass
class CFRSConfig:
    """Configuration for Cluster-First Route-Second (CF-RS) policy.

    Based on the geometric decomposition heuristic proposed by Fisher and Jaikumar (1981):
    "A generalized assignment heuristic for vehicle routing".

    This implementation focuses on angular partitioning of the customer set
    relative to a central depot, which is highly effective for radially
    distributed service areas.

    Attributes:
        seed: Random seed for reproducibility in the TSP routing phase.
            Ensures that the stochastic elements of the TSP solver (e.g., 2-opt
            initialization) yield consistent results.
        num_clusters: Number of angular sectors to partition nodes into.
            If set to 0, the policy defaults to using the number of vehicles
            available in the simulation day (`n_vehicles`).
        must_go: List of must-go strategy configuration files or dicts.
            Controls which bins are selected for collection on a given day.
        post_processing: List of post-processing operations (e.g., local search)
            to apply to the resulting tours for further refinement.
    """

    # Reproducibility seed for Routing phase
    seed: Optional[int] = None

    # Clustering granularity
    num_clusters: int = 0

    # Bin selection strategies (VRPP/WCVRP specific)
    must_go: Optional[List[MustGoConfig]] = None

    # Tour refinement strategies
    post_processing: Optional[List[PostProcessingConfig]] = None
