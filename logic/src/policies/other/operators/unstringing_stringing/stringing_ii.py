"""
Type II Stringing Operator.

Inserts node V_x back into the route with more complex reconnection.
This is the inverse operation of Type II Unstringing.
"""

from typing import List


def apply_type_ii_s(route: List[int], x: int, i: int, j: int, k: int, l: int) -> List[int]:
    """
    Apply Type II Stringing move.

    Inserts V_x between V_i and V_j.
    Deletes arcs: (V_i, V_{i+1}), (V_{l-1}, V_l), (V_j, V_{j+1}), (V_{k-1}, V_k)
    Inserts arcs: (V_i, V_x), (V_x, V_j), (V_l, V_{j+1}), (V_{k-1}, V_{l-1}), (V_{i+1}, V_k)

    Reverses sub-tours (V_{i+1}...V_{l-1}) and (V_l...V_j).

    Args:
        route: The tour as a list of node IDs.
        x: Node ID to insert (V_x).
        i: Index of node V_i (before insertion point).
        j: Index of node V_j (after insertion point).
        k: Index of node V_k (reconnection point).
        l: Index of node V_l (reconnection point).

    Returns:
        New route with V_x inserted and segments reconnected.

    Constraints:
        V_k != V_j, V_k != V_{j+1}, V_l != V_i, V_l != V_{i+1}
    """
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

    # Validate constraints
    val_k = route[k]
    val_j = route[j]
    val_l = route[l]
    val_i = route[i]

    if val_k == val_j or val_k == route[(j + 1) % n_work]:
        return route
    if val_l == val_i or val_l == route[(i + 1) % n_work]:
        return route

    # Rotate so V_i is at index 0
    pivot = i % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    # Find indices in rotated route
    try:
        j_new = rot_route.index(val_j)
        k_new = rot_route.index(val_k)
        l_new = rot_route.index(val_l)
    except ValueError:
        return route

    # Segments (assuming i < l < j < k in circular order):
    # S1: V_{i+1}...V_{l-1} -> indices [1, l_new)
    s1 = rot_route[1:l_new]

    # S2: V_l...V_j -> indices [l_new, j_new + 1)
    s2 = rot_route[l_new : j_new + 1]

    # S3: V_{j+1}...V_{k-1} -> indices [j_new + 1, k_new)
    s3 = rot_route[j_new + 1 : k_new]

    # Remainder: V_k...end -> indices [k_new, end]
    remainder = rot_route[k_new:]

    # Reconnection Logic:
    # V_i -> V_x -> V_j (reversed S2) -> V_{j+1} (S3) -> V_l
    # -> V_{l-1} (reversed S1) -> V_k
    # Sequence: V_i -> V_x -> s2_rev -> s3 -> s1_rev -> remainder
    new_rot = [rot_route[0]] + [x] + s2[::-1] + s3 + s1[::-1] + remainder

    # Restore depot
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route
