"""Shared helpers and utilities for route improvers.

This package provides stateless, reusable functions and classes for common
routing operations such as tour splitting, distance calculation, and
Thompson sampling for operator selection.

Attributes:
    ThompsonBandit (Type[ThompsonBandit]): Bandit implementation for operator selection.
    resolve_mandatory_nodes (Callable): Helper to find mandatory node IDs.
    split_tour (Callable): Splits a 0-separated tour into individual routes.
    assemble_tour (Callable): Assembles individual routes into a 0-separated tour.
    to_numpy (Callable): Safely converts data structures to numpy arrays.
    route_distance (Callable): Calculates the distance of a single route.
    tour_distance (Callable): Calculates the total distance of multiple routes.
    route_load (Callable): Calculates the total waste load of a route.
    upgrade_repair_op_to_profit (Callable): Converts repair operators to profit variants.

Example:
    >>> from logic.src.policies.route_improvement.common import split_tour, assemble_tour
    >>> routes = split_tour([0, 1, 2, 0, 3, 4, 0])
    >>> tour = assemble_tour(routes)
"""

from .bandit import ThompsonBandit
from .helpers import (
    assemble_tour,
    resolve_mandatory_nodes,
    route_distance,
    route_load,
    split_tour,
    to_numpy,
    tour_distance,
    upgrade_repair_op_to_profit,
)

__all__ = [
    "ThompsonBandit",
    "resolve_mandatory_nodes",
    "to_numpy",
    "split_tour",
    "assemble_tour",
    "route_distance",
    "tour_distance",
    "route_load",
    "upgrade_repair_op_to_profit",
]
