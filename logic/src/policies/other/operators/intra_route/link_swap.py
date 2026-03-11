"""
Link Swap Operator Module (Open-loop TSP).

A variant of 2-opt designed for open-loop tours where the start and end
depots are not connected.  Unlike standard 2-opt which reverses a segment
to maintain a closed cycle, link swap simply reconnects two edges without
the reversal constraint.

Given positions ``p_u`` and ``p_v`` (with ``p_u < p_v``), the operator
evaluates reconnecting ``route[p_u] → route[p_v]`` and
``route[p_u+1] → route[p_v+1]`` (reversing the middle segment).

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.intra_route.link_swap import link_swap
    >>> improved = link_swap(ls, r_idx=0, p_u=1, p_v=4)
"""

from typing import Any


def link_swap(ls: Any, r_idx: int, p_u: int, p_v: int) -> bool:
    """
    Link swap: open-loop 2-opt variant.

    Swaps the connections at two positions in a route.  For open-loop
    tours, this breaks edges (u, u_next) and (v, v_next) and reconnects
    as (u, v) and (u_next, v_next), reversing the segment between them.

    Args:
        ls: LocalSearch instance.
        r_idx: Route index.
        p_u: First edge position (lower).
        p_v: Second edge position (higher).

    Returns:
        bool: True if the link swap improved the solution.
    """
    route = ls.routes[r_idx]

    if p_u >= p_v:
        p_u, p_v = p_v, p_u

    if p_u + 1 >= p_v:
        return False
    if p_v >= len(route):
        return False

    u = route[p_u]
    u_next = route[p_u + 1]
    v = route[p_v]
    # Open-loop: v_next may be depot (0) or next node
    v_next = route[p_v + 1] if p_v + 1 < len(route) else 0

    old_cost = ls.d[u, u_next] + ls.d[v, v_next]
    new_cost = ls.d[u, v] + ls.d[u_next, v_next]

    delta = new_cost - old_cost

    if delta * ls.C < -1e-4:
        # Reverse the segment between p_u+1 and p_v (inclusive)
        route[p_u + 1 : p_v + 1] = route[p_u + 1 : p_v + 1][::-1]
        ls._update_map({r_idx})
        return True

    return False
