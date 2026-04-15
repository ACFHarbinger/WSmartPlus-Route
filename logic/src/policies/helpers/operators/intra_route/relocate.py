"""
Relocate and Relocate Chain (L3) Intra/Inter-Route Operators Module.

This module implements the relocate operator, which moves a single node
from its current position to a new position (after another node) in the
same or a different route. It also implements the relocate chain operator,
which extends the basic single-node relocate by allowing chains of k
consecutive nodes to be removed and reinserted at a different position.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.intra_route.relocate import move_relocate
    >>> improved = move_relocate(ls, u=1, v=2, r_u=0, p_u=1, r_v=0, p_v=5)
    >>> from logic.src.policies.helpers.operators.intra_route.relocate import relocate_chain
    >>> improved = relocate_chain(ls, r_src=0, pos_src=2, r_dst=0, pos_dst=5, chain_len=2)
"""

from typing import Any, List


def move_relocate(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """
    Relocate operator: move node u to a position after node v.

    Removes node u from route r_u and inserts it immediately after node v
    in route r_v. Can be inter-route or intra-route. Only applies the move
    if it improves total cost by a threshold margin.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Node to relocate.
        v: Node after which u will be inserted.
        r_u: Index of the route containing u.
        p_u: Position of u in route r_u.
        r_v: Index of the route containing v (target route).
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the relocation was applied (improving), False otherwise.
    """
    if r_u == r_v and (p_u == p_v + 1):
        return False
    dem_u = ls.waste.get(u, 0)

    if r_u != r_v and ls._get_load_cached(r_v) + dem_u > ls.Q:
        return False

    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]
    prev_u = route_u[p_u - 1] if p_u > 0 else 0
    next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

    delta = -ls.d[prev_u, u] - ls.d[u, next_u] + ls.d[prev_u, next_u]
    delta -= ls.d[v, v_next]
    delta += ls.d[v, u] + ls.d[u, v_next]

    if delta * ls.C < -1e-4:
        ls.routes[r_u].pop(p_u)
        if r_u == r_v and p_u < p_v:
            p_v -= 1
        ls.routes[r_v].insert(p_v + 1, u)
        ls._update_map({r_u, r_v})
        return True
    return False


def relocate_chain(ls: Any, r_src: int, pos_src: int, r_dst: int, pos_dst: int, chain_len: int = 1) -> bool:
    """
    Relocate L3: remove k consecutive nodes and reinsert elsewhere.

    Removes a chain of ``chain_len`` nodes starting at ``pos_src`` in
    route ``r_src`` and inserts them after ``pos_dst`` in route ``r_dst``.
    Works both intra-route and inter-route.

    Args:
        ls: LocalSearch instance.
        r_src: Source route index.
        pos_src: Position of the first node in the chain.
        r_dst: Destination route index.
        pos_dst: Insertion point (chain inserted after this position).
        chain_len: Number of consecutive nodes (1 or 2).

    Returns:
        bool: True if the relocation improved the solution.
    """
    if chain_len < 1:
        return False

    route_src = ls.routes[r_src]

    if pos_src + chain_len > len(route_src):
        return False

    chain = route_src[pos_src : pos_src + chain_len]
    dem_chain = sum(ls.waste.get(n, 0) for n in chain)

    # Skip trivial case: same route and inserting right where we removed
    if r_src == r_dst:
        if pos_dst >= pos_src and pos_dst < pos_src + chain_len:
            return False
    else:
        if ls._get_load_cached(r_dst) + dem_chain > ls.Q:
            return False

    route_dst = ls.routes[r_dst]

    # Removal cost
    prev_c = route_src[pos_src - 1] if pos_src > 0 else 0
    next_c = route_src[pos_src + chain_len] if pos_src + chain_len < len(route_src) else 0

    removal = _chain_edge_cost(ls.d, prev_c, chain, next_c)
    repair = ls.d[prev_c, next_c]

    # Insertion cost — compute on destination route
    # Adjust pos_dst for intra-route removal shift
    adj_pos_dst = pos_dst
    if r_src == r_dst and pos_src < pos_dst:
        adj_pos_dst -= chain_len

    # After removal, route_dst may have shifted (if same route)
    temp_dst = (
        list(route_dst)
        if r_src != r_dst
        else [n for i, n in enumerate(route_src) if i < pos_src or i >= pos_src + chain_len]
    )
    if adj_pos_dst >= len(temp_dst):
        adj_pos_dst = len(temp_dst) - 1

    v = temp_dst[adj_pos_dst] if adj_pos_dst >= 0 else 0
    v_next = temp_dst[adj_pos_dst + 1] if adj_pos_dst + 1 < len(temp_dst) else 0

    old_edge = ls.d[v, v_next]
    insertion = _chain_edge_cost(ls.d, v, chain, v_next)

    delta = (repair - removal) + (insertion - old_edge)

    if delta * ls.C < -1e-4:
        # Remove chain from source
        for _ in range(chain_len):
            route_src.pop(pos_src)

        # Adjust pos_dst for the removal
        insert_route = ls.routes[r_dst]
        ins_pos = pos_dst
        if r_src == r_dst and pos_src < pos_dst:
            ins_pos -= chain_len

        # Insert chain
        for idx, node in enumerate(chain):
            insert_route.insert(ins_pos + 1 + idx, node)

        ls._update_map({r_src, r_dst})
        return True

    return False


def move_or_opt(ls: Any, r_idx: int, pos: int, chain_len: int) -> bool:
    """
    Or-opt: relocate a chain of length chain_len to any other position in the same route.

    Wrapper around relocate_chain for intra-route moves.

    Args:
        ls: LocalSearch instance.
        r_idx: Index of the route.
        pos: Start position of the chain.
        chain_len: Length of the chain (1-3).

    Returns:
        bool: True if an improving move was found and applied.
    """
    route = ls.routes[r_idx]
    # Evaluate moving the chain to all other valid positions within the same route
    for target_pos in range(len(route)):
        # relocate_chain handles the 'same position' and 'invalid position' checks
        if relocate_chain(ls, r_src=r_idx, pos_src=pos, r_dst=r_idx, pos_dst=target_pos, chain_len=chain_len):
            return True
    return False


def _chain_edge_cost(d, prev_node: int, chain: List[int], next_node: int) -> float:
    """Cost of edges: prev → chain[0] → ... → chain[-1] → next."""
    if not chain:
        return d[prev_node, next_node]
    cost = d[prev_node, chain[0]]
    for i in range(len(chain) - 1):
        cost += d[chain[i], chain[i + 1]]
    cost += d[chain[-1], next_node]
    return cost
