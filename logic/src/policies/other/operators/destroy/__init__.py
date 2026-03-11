"""
Destroy Operators Package.

This package contains implementations of removal heuristics for ALNS, including
random, worst-case, cluster-based, Shaw (relatedness), string, route, neighbor,
historical, and sector removal.

Example:
    >>> from logic.src.policies.other.operators.destroy import random_removal
    >>> routes, removed = random_removal(routes, n=5)
"""

from ..unstringing_stringing import apply_type_i_us, apply_type_ii_us, apply_type_iii_us, apply_type_iv_us
from .cluster import cluster_removal
from .historical import historical_removal
from .neighbor import neighbor_removal
from .random import random_removal
from .route import route_removal
from .sector import sector_removal
from .shaw import shaw_removal
from .string import string_removal
from .worst import worst_removal

__all__ = [
    "random_removal",
    "worst_removal",
    "cluster_removal",
    "shaw_removal",
    "string_removal",
    "route_removal",
    "neighbor_removal",
    "historical_removal",
    "sector_removal",
    # Unstringing destroy
    "apply_type_i_us",
    "apply_type_ii_us",
    "apply_type_iii_us",
    "apply_type_iv_us",
]
