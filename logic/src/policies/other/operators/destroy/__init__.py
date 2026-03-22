"""
Destroy Operators Package.

This package contains implementations of removal heuristics for ALNS, including
random, worst-case, cluster-based, Shaw (relatedness), string, route, neighbor,
historical, and sector removal.

Also includes profit-based variants for VRPP problems: random_profit_removal,
cluster_profit_removal, worst_profit_removal, shaw_profit_removal,
neighbor_profit_removal, historical_profit_removal, string_profit_removal,
sector_profit_removal, and route_profit_removal.

Example:
    >>> from logic.src.policies.other.operators.destroy import random_removal
    >>> routes, removed = random_removal(routes, n=5)
    >>> from logic.src.policies.other.operators.destroy import worst_profit_removal
    >>> routes, removed = worst_profit_removal(routes, n=5, dist_matrix=d, wastes=w, R=1.0, C=1.0)
    >>> from logic.src.policies.other.operators.destroy import route_profit_removal
    >>> routes, removed = route_profit_removal(routes, dist_matrix=d, wastes=w, R=1.0, C=1.0)
"""

from ..unstringing_stringing import (
    apply_type_i_us,
    apply_type_i_us_profit,
    apply_type_ii_us,
    apply_type_ii_us_profit,
    apply_type_iii_us,
    apply_type_iii_us_profit,
    apply_type_iv_us,
    apply_type_iv_us_profit,
    stringing_insertion,
    stringing_profit_insertion,
    unstringing_profit_removal,
    unstringing_removal,
)
from .cluster import cluster_profit_removal, cluster_removal
from .historical import historical_profit_removal, historical_removal
from .neighbor import neighbor_profit_removal, neighbor_removal
from .random import random_profit_removal, random_removal
from .route import route_profit_removal, route_removal
from .sector import sector_profit_removal, sector_removal
from .shaw import shaw_profit_removal, shaw_removal
from .string import string_profit_removal, string_removal
from .worst import worst_profit_removal, worst_removal

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
    # Profit-based variants for VRPP
    "random_profit_removal",
    "cluster_profit_removal",
    "worst_profit_removal",
    "shaw_profit_removal",
    "neighbor_profit_removal",
    "historical_profit_removal",
    "string_profit_removal",
    "sector_profit_removal",
    "route_profit_removal",
    # Unstringing/stringing destroy/repair
    "apply_type_i_us",
    "apply_type_ii_us",
    "apply_type_iii_us",
    "apply_type_iv_us",
    "apply_type_i_us_profit",
    "apply_type_ii_us_profit",
    "apply_type_iii_us_profit",
    "apply_type_iv_us_profit",
    # Automated wrappers
    "unstringing_removal",
    "unstringing_profit_removal",
    "stringing_insertion",
    "stringing_profit_insertion",
]
