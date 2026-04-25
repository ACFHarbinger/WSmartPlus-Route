"""
Subramanian Inter-Route Operators Module.

This module implements the specific inter-route neighborhoods used in:
    Subramanian et al. (2013) "A hybrid algorithm for a class of vehicle routing problems"

Neighborhoods:
- Shift(2,0): Relocate 2 consecutive nodes.
- Swap(2,1): Swap 2 consecutive nodes with 1 node.
- Swap(2,2): Swap 2 consecutive nodes with 2 consecutive nodes.
- Cross: Exchange route suffixes (tails).

Attributes:
    shift_2_0: Shift(2,0) — relocate 2 consecutive nodes to another route.
    swap_2_1: Swap(2,1) — swap 2 consecutive nodes with 1 node.
    swap_2_2: Swap(2,2) — swap 2 consecutive nodes with 2 consecutive nodes.
    move_cross: Cross — exchange route suffixes (2-opt*).

Example:
    >>> from logic.src.policies.helpers.operators.inter_route_local_search.subramanian_neighborhoods import (
    ...     shift_2_0, swap_2_1, swap_2_2, move_cross,
    ... )
    >>> improved = shift_2_0(ls, r_src=0, pos_src=1, r_dst=1, pos_dst=2)
"""

from typing import Any

from .exchange_chain import exchange_k_0, exchange_k_h
from .k_opt_star import move_kopt_star


def shift_2_0(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Shift(2,0): Relocate 2 consecutive nodes from r_src to r_dst.
    Inserted after pos_dst.

    Args:
        ls: LocalSearch instance with routes and distance matrix.
        r_src: Source route index.
        pos_src: Position of the first node to relocate in r_src.
        r_dst: Destination route index.
        pos_dst: Insertion point in r_dst (nodes inserted after this position).

    Returns:
        bool: True if the move was applied (improving); False otherwise.
    """
    return exchange_k_0(ls, r_src, pos_src, r_dst, pos_dst, k=2)


def swap_2_1(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Swap(2,1): Swap 2 consecutive nodes from r_src with 1 node from r_dst.

    Args:
        ls: LocalSearch instance with routes and distance matrix.
        r_src: Source route index (contributes 2 nodes).
        pos_src: Position of the first source node in r_src.
        r_dst: Destination route index (contributes 1 node).
        pos_dst: Position of the destination node in r_dst.

    Returns:
        bool: True if the move was applied (improving); False otherwise.
    """
    return exchange_k_h(ls, r_src, pos_src, 2, r_dst, pos_dst, 1)


def swap_2_2(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int) -> bool:
    """
    Swap(2,2): Swap 2 consecutive nodes from r_src with 2 consecutive nodes from r_dst.

    Args:
        ls: LocalSearch instance with routes and distance matrix.
        r_src: Source route index.
        pos_src: Position of the first source node in r_src.
        r_dst: Destination route index.
        pos_dst: Position of the first destination node in r_dst.

    Returns:
        bool: True if the move was applied (improving); False otherwise.
    """
    return exchange_k_h(ls, r_src, pos_src, 2, r_dst, pos_dst, 2)


def move_cross(ls: Any, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    Cross: Exchange suffixes of routes r_u and r_v.
    Suffix of r_u starts after p_u; suffix of r_v starts after p_v.
    This is equivalent to 2-opt* (move_kopt_star with 2 cuts).

    Args:
        ls: LocalSearch instance with routes and distance matrix.
        u: Cut node in route r_u (last node of the head of r_u).
        v: Cut node in route r_v (last node of the head of r_v).
        r_u: Index of the first route.
        p_u: Position of u in r_u.
        r_v: Index of the second route.
        p_v: Position of v in r_v.

    Returns:
        bool: True if the suffix exchange was applied (improving); False otherwise.
    """
    return move_kopt_star(ls, [(u, r_u, p_u), (v, r_v, p_v)])
