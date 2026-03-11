"""
Type I Stringing Operator.

Inserts node V_x back into the route and reconnects by reversing two sub-tours.
This is the inverse operation of Type I Unstringing.
"""

from typing import List


def apply_type_i_s(route: List[int], x: int, i: int, j: int, k: int) -> List[int]:
    """
    Apply Type I Stringing move.

    Inserts V_x between V_i and V_j.
    Deletes arcs: (V_i, V_{i+1}), (V_j, V_{j-1}), (V_k, V_{k+1})
    Inserts arcs: (V_i, V_x), (V_x, V_j), (V_{i+1}, V_k), (V_{j+1}, V_{k+1})

    Reverses sub-tours (V_{i+1}...V_j) and (V_{j+1}...V_k).

    Args:
        route: The tour as a list of node IDs.
        x: Node ID to insert (V_x).
        i: Index of node V_i (before insertion point).
        j: Index of node V_j (after insertion point).
        k: Index of node V_k (reconnection point).

    Returns:
        New route with V_x inserted and segments reconnected.

    Constraints:
        V_k != V_i and V_k != V_j
    """
    # Defensive copy logic for route input
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

    # Validate constraints
    if route[k] == route[i] or route[k] == route[j]:
        return route

    # Rotate the route so V_i is at index 0
    pivot = i % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    # In rotated route:
    # Index 0: V_i
    # We need to find V_j and V_k

    val_j = route[j]
    val_k = route[k]

    # Find indices in rotated working route
    try:
        j_new = rot_route.index(val_j)
        k_new = rot_route.index(val_k)
    except ValueError:
        return route  # Fallback

    # Segments (assuming i < j < k in circular order):
    # S1: V_{i+1}...V_j -> indices [1, j_new + 1)
    s1 = rot_route[1 : j_new + 1]

    # S2: V_{j+1}...V_k -> indices [j_new + 1, k_new + 1)
    s2 = rot_route[j_new + 1 : k_new + 1]

    # Remainder: V_{k+1}...end -> indices [k_new + 1, end]
    remainder = rot_route[k_new + 1 :]

    # Reconnection Logic:
    # V_i -> V_x -> V_j (reversed s1) -> V_{i+1} -> V_k (reversed s2) -> V_{j+1} -> V_{k+1}
    # This means: V_i -> V_x -> s1_reversed -> s2_reversed -> remainder
    new_rot = [rot_route[0]] + [x] + s1[::-1] + s2[::-1] + remainder

    # Restore depot to front
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route
