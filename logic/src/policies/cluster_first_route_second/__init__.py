"""
Cluster-First Route-Second (CF-RS) Routing Policy.

This package implements the CF-RS algorithm, a variant of the Fisher and Jaikumar (1981)
heuristic. It partitions nodes into angular sectors centered at the depot (Clustering)
and solves the Traveling Salesman Problem (TSP) for each sector (Routing).

Modules:
    solver: Core implementation of the angular clustering and TSP routing.
    policy_cf_rs: Simulator adapter for the CF-RS algorithm.
"""

from .policy_cf_rs import ClusterFirstRouteSecondPolicy
from .solver import run_cf_rs

__all__ = ["ClusterFirstRouteSecondPolicy", "run_cf_rs"]
