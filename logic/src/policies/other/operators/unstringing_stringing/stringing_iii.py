"""
Type III Stringing Operator.

Inserts node V_x and reverses almost the entire sequence.
This is the inverse of Stringing Type I, exploring other promising regions.
"""

from typing import List


def apply_type_iii_s(route: List[int], x: int, i: int, j: int, k: int) -> List[int]:
    """
    Apply Type III Stringing move.

    Inserts V_x between V_i and V_j, reversing almost the entire sequence.
    Deletes arcs: (V_{i-1}, V_i), (V_{j-1}, V_j), (V_{k-1}, V_k)
    Inserts arcs: (V_i, V_x), (V_x, V_j), (V_k, V_{j-1}), (V_{k-1}, V_{i-1})

    Reverses sub-tours (V_i...V_{j-1}) and (V_k...V_{i-1}).

    Args:
        route: The tour as a list of node IDs.
        x: Node ID to insert (V_x).
        i: Index of node V_i (start of first reversal).
        j: Index of node V_j (end of first reversal + 1).
        k: Index of node V_k (start of second reversal).

    Returns:
        New route with V_x inserted and segments reversed.

    Constraints:
        V_k != V_i and V_k != V_j
    """
    n = len(route)
    is_closed = n > 1 and route[0] == route[-1]
    work_route = route[:-1] if is_closed else route[:]
    n_work = len(work_route)

    # Validate constraints
    if route[k] == route[i] or route[k] == route[j]:
        return route

    # Rotate the route so V_{i-1} is at the end (V_i is at index 0)
    pivot = i % n_work
    rot_route = work_route[pivot:] + work_route[:pivot]

    val_j = route[j]
    val_k = route[k]

    # Find indices in rotated working route
    try:
        j_new = rot_route.index(val_j)
        k_new = rot_route.index(val_k)
    except ValueError:
        return route

    # Segments:
    # S1: V_i...V_{j-1} -> indices [0, j_new)
    s1 = rot_route[0:j_new]

    # S2: V_j...V_{k-1} -> indices [j_new, k_new)
    s2 = rot_route[j_new:k_new]

    # S3: V_k...V_{i-1} -> indices [k_new, end]
    s3 = rot_route[k_new:]

    # Reconnection Logic:
    # The operation reverses S1 and S3
    # New sequence: s1_rev[0] (V_i) -> V_x -> V_j (s2[0]) -> ... -> s2_end
    # -> s3_rev -> s1_rev[1:]
    # Result: V_i -> V_x -> s2 -> s3_rev -> s1_rev[1:]
    new_rot = [s1[0]] + [x] + s2 + s3[::-1] + s1[1:][::-1]

    # Restore depot to front
    if 0 in new_rot:
        zero_idx = new_rot.index(0)
        final_route = new_rot[zero_idx:] + new_rot[:zero_idx]
    else:
        final_route = new_rot

    if is_closed:
        final_route.append(final_route[0])

    return final_route
