"""
Adaptive mutation and neighborhood operators for SANS-style optimization.

Implements specialized local search operators including cross-exchange,
relocation, Or-opt moves, and 2-opt swaps tailored for the SANS (Simulated
Annealing with Adaptive Neighborhood Selection) policy variant.

This module acts as a facade, re-exporting operators from specialized sub-modules.

Attributes:
    get_neighbors: Generate neighbors using all basic operators.
    get_2opt_neighbors: Generate neighbors using 2-opt swaps.
    relocate_within_route: Move a bin within the same route.
    cross_exchange: Swap bins between two routes.
    or_opt_move: Re-insert a bin at a different position in the same route.
    move_between_routes: Move a bin from one route to another.
    insert_bin_in_route: Insert a bin into a route.
    mutate_route_by_swapping_bins: Swap bins within a route.
    remove_bins_from_route: Remove bins from a route and add them to the removed pool.
    move_n_route_random: Move n random bins from one route to another.
    swap_n_route_random: Swap n random bins between two distinct routes.
    remove_n_bins_random: Remove n random bins from routes and add them to the removed pool.
    add_n_bins_random: Add n random bins from the removed pool back into routes.
    add_route_with_removed_bins_random: Create a new route from random bins in the removed pool.
    move_n_route_consecutive: Move a sequence of consecutive bins from one route to another.
    swap_n_route_consecutive: Swap a sequence of consecutive bins between two distinct routes.
    remove_n_bins_consecutive: Remove a sequence of consecutive bins from routes and add them to the removed pool.
    add_n_bins_consecutive: Add a sequence of consecutive bins from the removed pool back into routes.
    add_route_with_removed_bins_consecutive: Create a new route from a sequence of consecutive bins in the removed pool.

Example:
    >>> import random
    >>> routes = [[0, 1, 2, 3, 0], [0, 4, 5, 6, 0]]
    >>> bins_cannot_removed = {1, 4}
    >>> rng = random.Random()
    >>> removed_bins = set()
    >>> new_routes = remove_bins_from_route(routes, bins_cannot_removed, rng, num_bins=1)
    >>> new_routes
    [[0, 2, 3, 0], [0, 4, 5, 6, 0]]
"""

from .sans_neighborhoods import (
    cross_exchange,
    get_2opt_neighbors,
    get_neighbors,
    insert_bin_in_route,
    move_between_routes,
    mutate_route_by_swapping_bins,
    or_opt_move,
    relocate_within_route,
)
from .sans_perturbations import (
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    move_n_route_consecutive,
    move_n_route_random,
    remove_bins_from_route,
    remove_n_bins_consecutive,
    remove_n_bins_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)

__all__ = [
    "get_neighbors",
    "get_2opt_neighbors",
    "relocate_within_route",
    "cross_exchange",
    "or_opt_move",
    "move_between_routes",
    "insert_bin_in_route",
    "mutate_route_by_swapping_bins",
    "remove_bins_from_route",
    "move_n_route_random",
    "swap_n_route_random",
    "remove_n_bins_random",
    "add_n_bins_random",
    "add_route_with_removed_bins_random",
    "move_n_route_consecutive",
    "swap_n_route_consecutive",
    "remove_n_bins_consecutive",
    "add_n_bins_consecutive",
    "add_route_with_removed_bins_consecutive",
]
