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
