"""
Route-based local search operators for HGS.
"""

import random


def move_swap_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """SWAP* inter-route operator: swap nodes u and v between different routes.

    Removes u from route r_u and v from route r_v, then reinserts each node
    into the other route at the best position. Only applies the move if it
    improves the total cost by a threshold margin.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Node to remove from route r_u.
        v: Node to remove from route r_v.
        r_u: Index of the route containing u.
        p_u: Position of u in route r_u.
        r_v: Index of the route containing v.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the swap was applied (improving), False otherwise.
    """
    dem_u = ls.demands.get(u, 0)
    dem_v = ls.demands.get(v, 0)

    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]

    prev_u = route_u[p_u - 1] if p_u > 0 else 0
    next_u = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    gain_rem_u = ls.d[prev_u, u] + ls.d[u, next_u] - ls.d[prev_u, next_u]

    prev_v = route_v[p_v - 1] if p_v > 0 else 0
    next_v = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0
    gain_rem_v = ls.d[prev_v, v] + ls.d[v, next_v] - ls.d[prev_v, next_v]

    temp_rv = route_v[:]
    temp_rv.pop(p_v)
    if ls._calc_load_fresh(temp_rv) + dem_u > ls.Q:
        return False

    best_delta_u = float("inf")
    best_pos_u = -1

    for pos in range(len(temp_rv) + 1):
        prev = temp_rv[pos - 1] if pos > 0 else 0
        nxt = temp_rv[pos] if pos < len(temp_rv) else 0
        delta = ls.d[prev, u] + ls.d[u, nxt] - ls.d[prev, nxt]
        if delta < best_delta_u:
            best_delta_u = delta
            best_pos_u = pos

    temp_ru = route_u[:]
    temp_ru.pop(p_u)
    if ls._calc_load_fresh(temp_ru) + dem_v > ls.Q:
        return False

    best_delta_v = float("inf")
    best_pos_v = -1

    for pos in range(len(temp_ru) + 1):
        prev = temp_ru[pos - 1] if pos > 0 else 0
        nxt = temp_ru[pos] if pos < len(temp_ru) else 0
        delta = ls.d[prev, v] + ls.d[v, nxt] - ls.d[prev, nxt]
        if delta < best_delta_v:
            best_delta_v = delta
            best_pos_v = pos

    total_delta = -gain_rem_u - gain_rem_v + best_delta_u + best_delta_v

    if total_delta * ls.C < -1e-4:
        ls.routes[r_u] = temp_ru
        ls.routes[r_u].insert(best_pos_v, v)
        ls.routes[r_v] = temp_rv
        ls.routes[r_v].insert(best_pos_u, u)
        ls._update_map({r_u, r_v})
        return True
    return False


def move_2opt_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """2-opt* inter-route operator: exchange tails between two routes.

    Cuts route r_u after node u and route r_v after node v, then swaps
    the tail segments. Only applies the move if it improves total cost.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Cut point node in route r_u.
        v: Cut point node in route r_v.
        r_u: Index of first route.
        p_u: Position of u in route r_u.
        r_v: Index of second route.
        p_v: Position of v in route r_v.

    Returns:
        bool: True if the exchange was applied (improving), False otherwise.
    """
    route_u = ls.routes[r_u]
    route_v = ls.routes[r_v]

    tail_u = route_u[p_u + 1 :]
    tail_v = route_v[p_v + 1 :]

    l_head_u = ls._calc_load_fresh(route_u[: p_u + 1])
    l_head_v = ls._calc_load_fresh(route_v[: p_v + 1])
    l_tail_u = ls._get_load_cached(r_u) - l_head_u
    l_tail_v = ls._get_load_cached(r_v) - l_head_v

    if l_head_u + l_tail_v > ls.Q or l_head_v + l_tail_u > ls.Q:
        return False

    u_next = route_u[p_u + 1] if p_u < len(route_u) - 1 else 0
    v_next = route_v[p_v + 1] if p_v < len(route_v) - 1 else 0

    delta = -ls.d[u, u_next] - ls.d[v, v_next] + ls.d[u, v_next] + ls.d[v, u_next]

    if delta * ls.C < -1e-4:
        new_ru = route_u[: p_u + 1] + tail_v
        new_rv = route_v[: p_v + 1] + tail_u
        ls.routes[r_u] = new_ru
        ls.routes[r_v] = new_rv
        ls._update_map({r_u, r_v})
        return True
    return False


def move_2opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """2-opt intra-route operator: reverse a segment within a route.

    Reverses the segment between positions p_u+1 and p_v (inclusive)
    in route r_u. Only applies the move if it reduces total cost.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: Start node of the segment (edge u->next_u broken).
        v: End node of the segment (edge v->next_v broken).
        r_u: Index of the route (must equal r_v for intra-route).
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.

    Returns:
        bool: True if the reversal was applied (improving), False otherwise.
    """
    if p_u >= p_v:
        return False
    if p_u + 1 == p_v:
        return False

    route = ls.routes[r_u]
    u_next = route[p_u + 1]
    v_next = route[p_v + 1] if p_v < len(route) - 1 else 0

    delta = -ls.d[u, u_next] - ls.d[v, v_next] + ls.d[u, v] + ls.d[u_next, v_next]

    if delta * ls.C < -1e-4:
        segment = route[p_u + 1 : p_v + 1]
        route[p_u + 1 : p_v + 1] = segment[::-1]
        ls._update_map({r_u})
        return True
    return False


def move_3opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool:
    """3-opt intra-route operator: reconnect three segments within a route.

    Randomly selects a third cut point and evaluates all 3-opt reconnection
    patterns. Applies the best improving move if found.

    Args:
        ls: LocalSearch instance containing routes and distance matrix.
        u: First cut point node.
        v: Second cut point node.
        r_u: Index of the route.
        p_u: Position of u in the route.
        r_v: Index of the route (unused but required for signature).
        p_v: Position of v in the route.

    Returns:
        bool: True if a 3-opt move was applied (improving), False otherwise.
    """
    route = ls.routes[r_u]
    if len(route) < 4:
        return False
    if p_u > p_v:
        p_u, p_v = p_v, p_u
        u, v = v, u

    for _ in range(5):
        p_w = random.randint(0, len(route) - 1)
        if p_w in {p_u, p_v, p_u + 1, p_v + 1, p_u - 1, p_v - 1}:
            continue

        idx = sorted([p_u, p_v, p_w])
        i, j, k = idx[0], idx[1], idx[2]

        A, B = route[i], route[i + 1] if i < len(route) - 1 else 0
        C, D = route[j], route[j + 1] if j < len(route) - 1 else 0
        E, F = route[k], route[k + 1] if k < len(route) - 1 else 0

        d_base = ls.d[A, B] + ls.d[C, D] + ls.d[E, F]

        g4 = d_base - (ls.d[A, C] + ls.d[B, E] + ls.d[D, F])
        g5 = d_base - (ls.d[A, D] + ls.d[E, B] + ls.d[C, F])
        g6 = d_base - (ls.d[A, D] + ls.d[E, C] + ls.d[B, F])
        g7 = d_base - (ls.d[A, E] + ls.d[D, B] + ls.d[C, F])

        gains = [g4, g5, g6, g7]
        best_g = max(gains)
        if best_g * ls.C > 1e-4:
            case = gains.index(best_g)
            s1 = route[: i + 1]
            s2 = route[i + 1 : j + 1]
            s3 = route[j + 1 : k + 1]
            s4 = route[k + 1 :]

            if case == 0:
                route[:] = s1 + s2[::-1] + s3[::-1] + s4
            elif case == 1:
                route[:] = s1 + s3 + s2 + s4
            elif case == 2:
                route[:] = s1 + s3 + s2[::-1] + s4
            elif case == 3:
                route[:] = s1 + s3[::-1] + s2 + s4

            ls._update_map({r_u})
            return True
    return False
