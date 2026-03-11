"""
GENI (Generalized Insertion) Operator Module.

Implements GENI Type I and Type II moves that insert a node between two
*non-adjacent* nodes in the tour, followed by localized edge re-optimization.

- **Type I**: Insert node ``u`` between non-adjacent ``v_i`` and ``v_j``,
  reconnecting with a single bypass.
- **Type II**: Insert node ``u`` between ``v_i`` and ``v_j`` while
  simultaneously reversing the segment between them.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.intra_route.geni import geni_insert
    >>> improved = geni_insert(ls, node=5, r_idx=0, neighborhood_size=5)
"""

from typing import Any, List, Optional, Tuple


def geni_insert(ls: Any, node: int, r_idx: int, neighborhood_size: int = 5) -> bool:
    """
    GENI insertion: insert a removed node using Type I/II moves.

    Finds the best insertion among the node's nearest neighbors in the
    given route, evaluating both Type I (direct) and Type II (reversal)
    reconnections.

    Args:
        ls: LocalSearch instance.
        node: Node to insert (must not already be in a route).
        r_idx: Target route index.
        neighborhood_size: Max neighbors to consider around each candidate.

    Returns:
        bool: True if an improving insertion was found and applied.
    """
    route = ls.routes[r_idx]
    if len(route) < 3:
        return False

    dem = ls.waste.get(node, 0)
    if ls._get_load_cached(r_idx) + dem > ls.Q:
        return False

    # Find nearest nodes to `node` that are in this route
    neighbors = _get_nearest_in_route(ls.d, node, route, neighborhood_size)

    best_gain = 0.0
    best_move: Optional[Tuple[str, int, int]] = None  # (type, pos_i, pos_j)
    for v_i in neighbors:
        pos_i = route.index(v_i)
        for v_j in neighbors:
            if v_i == v_j:
                continue
            pos_j = route.index(v_j)
            if abs(pos_i - pos_j) <= 1:
                continue  # Adjacent — not a GENI move

            # Type I: insert node between v_i and v_j, bypass intermediate
            gain_i = _evaluate_type_i(ls, route, node, pos_i, pos_j)
            if gain_i > best_gain:
                best_gain = gain_i
                best_move = ("I", pos_i, pos_j)

            # Type II: insert node, reverse segment between positions
            gain_ii = _evaluate_type_ii(ls, route, node, pos_i, pos_j)
            if gain_ii > best_gain:
                best_gain = gain_ii
                best_move = ("II", pos_i, pos_j)

    if best_move is not None and best_gain * ls.C > 1e-4:
        move_type, pi, pj = best_move
        if move_type == "I":
            _apply_type_i(ls, route, node, pi, pj, r_idx)
        else:
            _apply_type_ii(ls, route, node, pi, pj, r_idx)
        return True

    return False


def _get_nearest_in_route(d, node: int, route: List[int], k: int) -> List[int]:
    """Return up to k nearest route nodes to `node` by distance."""
    dists = [(d[node, v], v) for v in route]
    dists.sort()
    return [v for _, v in dists[:k]]


def _evaluate_type_i(ls: Any, route: List[int], node: int, pi: int, pj: int) -> float:
    """
    Type I gain: insert node, connect v_i→node→v_j, bypass v_i+1 to v_j-1.

    Cost saved: d(v_i, v_i+1) + d(v_j-1, v_j)
    Cost added: d(v_i, node) + d(node, v_j)
    Net connection of bypassed ends: d(v_i+1, v_j-1) handled by adjacency.
    """
    if pi > pj:
        pi, pj = pj, pi

    v_i = route[pi]
    v_i_next = route[pi + 1] if pi + 1 < len(route) else 0
    v_j = route[pj]
    v_j_prev = route[pj - 1] if pj > 0 else 0

    cost_removed = ls.d[v_i, v_i_next] + ls.d[v_j_prev, v_j]
    cost_added = ls.d[v_i, node] + ls.d[node, v_j] + ls.d[v_i_next, v_j_prev]

    return cost_removed - cost_added


def _evaluate_type_ii(ls: Any, route: List[int], node: int, pi: int, pj: int) -> float:
    """
    Type II gain: insert node between v_i and v_j, reverse the segment.

    Similar to Type I but the segment [pi+1..pj-1] is reversed, so
    connections become v_i→node→v_j, with reversed internal segment.
    """
    if pi > pj:
        pi, pj = pj, pi

    v_i = route[pi]
    v_i_next = route[pi + 1] if pi + 1 < len(route) else 0
    v_j = route[pj]
    v_j_prev = route[pj - 1] if pj > 0 else 0

    cost_removed = ls.d[v_i, v_i_next] + ls.d[v_j_prev, v_j]
    # After reversal: v_i → node → v_j_prev(now first of reversed) ... v_i_next(now last) → v_j
    cost_added = ls.d[v_i, node] + ls.d[node, v_j_prev] + ls.d[v_i_next, v_j]

    return cost_removed - cost_added


def _apply_type_i(ls: Any, route: List[int], node: int, pi: int, pj: int, r_idx: int) -> None:
    """Apply Type I GENI move."""
    if pi > pj:
        pi, pj = pj, pi
    # Remove segment between pi and pj, reconnect v_i+1..v_j-1
    # Insert node between v_i and v_j
    mid = route[pi + 1 : pj]
    new_route = route[: pi + 1] + [node] + route[pj:]
    # Reinsert the middle segment after v_j
    insert_pos = pi + 2  # after node
    for idx, m in enumerate(mid):
        new_route.insert(insert_pos + idx, m)
    ls.routes[r_idx] = new_route
    ls._update_map({r_idx})


def _apply_type_ii(ls: Any, route: List[int], node: int, pi: int, pj: int, r_idx: int) -> None:
    """Apply Type II GENI move (with segment reversal)."""
    if pi > pj:
        pi, pj = pj, pi
    mid = route[pi + 1 : pj]
    mid_reversed = mid[::-1]
    new_route = route[: pi + 1] + [node] + mid_reversed + route[pj:]
    ls.routes[r_idx] = new_route
    ls._update_map({r_idx})
