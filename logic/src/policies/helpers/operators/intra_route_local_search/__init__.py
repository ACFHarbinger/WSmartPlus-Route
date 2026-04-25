"""
Intra-route Local Search Operators.

This package contains operators that perform moves within a single route,
or moves where the source and destination may be the same or different routes.

Attributes:
    move_relocate: Relocate a single node to another position.
    move_swap: Swap two nodes within or between routes.
    move_2opt_intra: 2-opt intra-route edge reversal.
    move_3opt_intra: 3-opt intra-route triple-edge reconnection.
    move_kopt_intra: General k-opt intra-route operator.
    move_or_opt: Or-opt chain relocation within a route.
    or_opt_route: Or-opt applied to a plain route list.
    three_opt_route: 3-opt applied to a plain route list.
    two_opt_route: 2-opt applied to a plain route list.
    k_permutation: Re-order k consecutive nodes to find a cheaper arrangement.
    three_permutation: 3-permutation convenience wrapper.
    relocate_chain: Relocate a chain of consecutive nodes.
    apply_intra_route_cross_exchange: Intra-route CROSS-exchange segment swap.

Example:
    >>> from logic.src.policies.helpers.operators.intra_route_local_search import (
    ...     move_2opt_intra, move_relocate, move_or_opt,
    ... )
    >>> improved = move_2opt_intra(ls, u=3, v=7, r_u=0, p_u=2, r_v=0, p_v=5)
"""

from .cross_exchange import apply_intra_route_cross_exchange
from .k_opt import move_2opt_intra, move_3opt_intra, move_kopt_intra, three_opt_route, two_opt_route
from .k_permutation import k_permutation, three_permutation
from .relocate import move_or_opt, move_relocate, or_opt_route, relocate_chain
from .swap import move_swap

__all__ = [
    "move_relocate",
    "move_swap",
    "move_2opt_intra",
    "move_3opt_intra",
    "move_kopt_intra",
    "move_or_opt",
    "or_opt_route",
    "three_opt_route",
    "two_opt_route",
    "k_permutation",
    "three_permutation",
    "relocate_chain",
    "apply_intra_route_cross_exchange",
]
