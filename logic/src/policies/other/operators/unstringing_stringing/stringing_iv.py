"""
Type IV Stringing Operator.

Most complex stringing operator, inverse of Type II Unstringing.
Involves four neighbor nodes and multiple reversals.
"""

from typing import List


def apply_type_iv_s(route: List[int], x: int, i: int, j: int, k: int, l: int) -> List[int]:
    """
    Apply Type IV Stringing move.

    Inserts V_x between V_i and V_j with complex reconnection.
    Deletes arcs: (V_{i-1}, V_i), (V_l, V_{l+1}), (V_{j-1}, V_j), (V_k, V_{k+1})
    Inserts arcs: (V_i, V_x), (V_x, V_j), (V_{i-1}, V_l), (V_{l+1}, V_{k+1}), (V_k, V_{j-1})

    Reverses sub-tours (V_i...V_l) and (V_{l+1}...V_{j-1}).

    Args:
        route: The tour as a list of node IDs.
        x: Node ID to insert (V_x).
        i: Index of node V_i (start of first reversal).
        j: Index of node V_j (end of second reversal + 1).
        k: Index of node V_k (reconnection point).
        l: Index of node V_l (split point between reversals).

    Returns:
        New route with V_x inserted and segments reversed.

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

    # Rotate so V_{i-1} is at the end (V_i at index 0)
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
    # S1: V_i...V_l -> indices [0, l_new + 1)
    s1 = rot_route[0 : l_new + 1]

    # S2: V_{l+1}...V_{j-1} -> indices [l_new + 1, j_new)
    s2 = rot_route[l_new + 1 : j_new]

    # S3: V_j...V_k -> indices [j_new, k_new + 1)
    s3 = rot_route[j_new : k_new + 1]

    # Remainder: V_{k+1}...V_{i-1} -> indices [k_new + 1, end]
    remainder = rot_route[k_new + 1 :]

    # Reconnection Logic:
    # V_i -> V_x -> V_j (s3[0]) -> ... -> s3_end (but we reverse parts)
    # New sequence: s1_rev[0] (V_i) -> V_x -> s3[0] (V_j) -> s3[1:] -> s2_rev -> s1_rev[1:] -> remainder
    # Simplifying: [V_i] + [V_x] + s3 + s2_rev + s1_rev[1:] + remainder
    # But we need: V_i -> V_x -> V_j (with reversals of S1 and S2)
    # Result: s1_rev[0] + V_x + s3[0] + s2_rev + s1_rev[1:] + s3[1:] + remainder

    # According to spec:
    # Sub tours (V_i...V_l) and (V_{l+1}...V_{j-1}) are reversed
    # New order should be: V_i -> V_x -> V_j -> ... following the arc pattern

    # Let me recalculate based on the arc pattern:
    # Inserts: (V_i, V_x), (V_x, V_j), (V_{i-1}, V_l), (V_{l+1}, V_{k+1}), (V_k, V_{j-1})
    # This means: ... -> V_{i-1} -> V_l (rev of s1) -> V_i -> V_x -> V_j (start of s3)
    # -> ... V_k -> V_{j-1} (rev of s2) -> V_{l+1} -> V_{k+1} -> ...

    new_rot = [s1[0]] + [x] + [s3[0]] + s2[::-1] + s1[1:][::-1] + s3[1:] + remainder

    # Restore depot to front
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route
