"""
Exchange operators for VRP local search.

This module contains advanced inter-route and intra-route operators:
- Or-opt: Relocate chains of consecutive nodes
- Cross-Exchange: Swap segments between routes (λ-interchange)
- Ejection Chain: Compound displacement for fleet reduction
"""

from typing import Any, List, Optional, Tuple


def move_or_opt(
    ls: Any,
    node: int,
    chain_len: int,
    r_idx: int,
    pos: int,
) -> bool:
    """
    Or-opt: Relocate a chain of consecutive nodes to the best position.

    Moves a sequence of 1-3 consecutive customers to another position,
    either within the same route or to a different route. Particularly
    effective when customers are geographically clustered.

    Args:
        ls: LocalSearch instance with routes, distance matrix, demands, etc.
        node: Starting node of the chain.
        chain_len: Length of chain to move (1, 2, or 3).
        r_idx: Route index containing the chain.
        pos: Position of the starting node in the route.

    Returns:
        True if an improving move was found and applied.
    """
    route = ls.routes[r_idx]
    if pos + chain_len > len(route):
        return False

    # Extract chain
    chain = route[pos : pos + chain_len]
    chain_demand = sum(ls.demands.get(n, 0) for n in chain)

    # Calculate removal cost
    prev_node = route[pos - 1] if pos > 0 else 0
    next_node = route[pos + chain_len] if pos + chain_len < len(route) else 0
    removal_gain = ls.d[prev_node, chain[0]] + ls.d[chain[-1], next_node] - ls.d[prev_node, next_node]

    best_delta = 0.0
    best_insertion = None  # (route_idx, insert_pos)

    # Try all positions in all routes
    for target_r_idx, target_route in enumerate(ls.routes):
        # Check capacity
        if target_r_idx != r_idx:
            target_load = ls._calc_load_fresh(target_route)
            if target_load + chain_demand > ls.Q:
                continue

        # Try all insertion positions
        for insert_pos in range(len(target_route) + 1):
            # Skip positions that would just reinsert at same spot
            if target_r_idx == r_idx and insert_pos in range(pos, pos + chain_len + 1):
                continue

            ins_prev = target_route[insert_pos - 1] if insert_pos > 0 else 0
            ins_next = target_route[insert_pos] if insert_pos < len(target_route) else 0

            # Adjust for removal if same route
            if target_r_idx == r_idx:
                if insert_pos > pos + chain_len:
                    # Actually insert at insert_pos - chain_len after removal
                    temp_route = route[:pos] + route[pos + chain_len :]
                    adj_pos = insert_pos - chain_len
                    ins_prev = temp_route[adj_pos - 1] if adj_pos > 0 else 0
                    ins_next = temp_route[adj_pos] if adj_pos < len(temp_route) else 0

            insertion_cost = ls.d[ins_prev, chain[0]] + ls.d[chain[-1], ins_next] - ls.d[ins_prev, ins_next]

            delta = insertion_cost - removal_gain

            if delta < best_delta - 1e-6:
                best_delta = delta
                best_insertion = (target_r_idx, insert_pos)

    # Apply best move
    if best_insertion is not None:
        target_r, ins_pos = best_insertion

        # Remove chain from original route
        del ls.routes[r_idx][pos : pos + chain_len]

        # Adjust insertion position if same route and after removal point
        if target_r == r_idx and ins_pos > pos + chain_len:
            ins_pos -= chain_len

        # Insert chain
        for i, n in enumerate(chain):
            ls.routes[target_r].insert(ins_pos + i, n)

        ls._update_map({r_idx, target_r})
        return True

    return False


def cross_exchange(
    ls: Any,
    r_a: int,
    seg_a_start: int,
    seg_a_len: int,
    r_b: int,
    seg_b_start: int,
    seg_b_len: int,
) -> bool:
    """
    Cross-Exchange (λ-interchange): Swap segments between two routes.

    Exchanges a segment of customers from Route A with a segment from Route B.
    The internal order of both segments is preserved.

    Args:
        ls: LocalSearch instance.
        r_a: Index of first route.
        seg_a_start: Start position of segment in Route A.
        seg_a_len: Length of segment in Route A.
        r_b: Index of second route.
        seg_b_start: Start position of segment in Route B.
        seg_b_len: Length of segment in Route B.

    Returns:
        True if an improving exchange was applied.
    """
    if r_a == r_b:
        return False

    route_a = ls.routes[r_a]
    route_b = ls.routes[r_b]

    if seg_a_start + seg_a_len > len(route_a):
        return False
    if seg_b_start + seg_b_len > len(route_b):
        return False

    # Extract segments
    seg_a = route_a[seg_a_start : seg_a_start + seg_a_len]
    seg_b = route_b[seg_b_start : seg_b_start + seg_b_len]

    # Check capacity feasibility
    demand_a = sum(ls.demands.get(n, 0) for n in seg_a)
    demand_b = sum(ls.demands.get(n, 0) for n in seg_b)

    load_a = ls._calc_load_fresh(route_a)
    load_b = ls._calc_load_fresh(route_b)

    new_load_a = load_a - demand_a + demand_b
    new_load_b = load_b - demand_b + demand_a

    if new_load_a > ls.Q or new_load_b > ls.Q:
        return False

    # Calculate delta
    # Route A: remove seg_a, insert seg_b
    a_prev = route_a[seg_a_start - 1] if seg_a_start > 0 else 0
    a_next = route_a[seg_a_start + seg_a_len] if seg_a_start + seg_a_len < len(route_a) else 0

    removal_a = ls.d[a_prev, seg_a[0]] + ls.d[seg_a[-1], a_next]
    insertion_a = ls.d[a_prev, seg_b[0]] + ls.d[seg_b[-1], a_next] if seg_b else ls.d[a_prev, a_next]

    # Route B: remove seg_b, insert seg_a
    b_prev = route_b[seg_b_start - 1] if seg_b_start > 0 else 0
    b_next = route_b[seg_b_start + seg_b_len] if seg_b_start + seg_b_len < len(route_b) else 0

    removal_b = ls.d[b_prev, seg_b[0]] + ls.d[seg_b[-1], b_next]
    insertion_b = ls.d[b_prev, seg_a[0]] + ls.d[seg_a[-1], b_next] if seg_a else ls.d[b_prev, b_next]

    delta = (insertion_a - removal_a) + (insertion_b - removal_b)

    if delta * ls.C < -1e-4:
        # Apply exchange
        new_route_a = route_a[:seg_a_start] + seg_b + route_a[seg_a_start + seg_a_len :]
        new_route_b = route_b[:seg_b_start] + seg_a + route_b[seg_b_start + seg_b_len :]

        ls.routes[r_a] = new_route_a
        ls.routes[r_b] = new_route_b
        ls._update_map({r_a, r_b})
        return True

    return False


def ejection_chain(
    ls: Any,
    source_route: int,
    max_depth: int = 5,
) -> bool:
    """
    Ejection Chain: Compound displacement for fleet reduction.

    Attempts to empty a route by ejecting its customers into other routes.
    When a target route is full, it triggers a chain of ejections.

    This operator is primarily used for fleet size minimization.

    Args:
        ls: LocalSearch instance.
        source_route: Index of route to empty.
        max_depth: Maximum chain depth before giving up.

    Returns:
        True if the source route was successfully emptied.
    """
    if source_route >= len(ls.routes) or not ls.routes[source_route]:
        return False

    # Try to empty the source route
    route = ls.routes[source_route]
    nodes_to_eject = route[:]

    # Track successful insertions
    ejection_log: List[Tuple[int, int, int]] = []  # (node, target_route, position)

    for node in nodes_to_eject:
        inserted = _try_insert_with_chain(ls, node, source_route, max_depth, ejection_log)
        if not inserted:
            # Rollback all insertions
            _rollback_ejections(ls, ejection_log, source_route)
            return False

    # All nodes successfully ejected
    if ls.routes[source_route]:
        ls.routes[source_route] = []
    ls._update_map(set(range(len(ls.routes))))
    return True


def _try_insert_with_chain(
    ls: Any,
    node: int,
    excluded_route: int,
    depth: int,
    log: List[Tuple[int, int, int]],
) -> bool:
    """Try to insert a node, potentially triggering chain ejections."""
    if depth <= 0:
        return False

    node_demand = ls.demands.get(node, 0)

    # Find best insertion
    best_cost = float("inf")
    best_insertion: Optional[Tuple[int, int]] = None

    for r_idx, route in enumerate(ls.routes):
        if r_idx == excluded_route:
            continue

        load = ls._calc_load_fresh(route)
        if load + node_demand > ls.Q:
            continue

        for pos in range(len(route) + 1):
            prev = route[pos - 1] if pos > 0 else 0
            nxt = route[pos] if pos < len(route) else 0
            cost = ls.d[prev, node] + ls.d[node, nxt] - ls.d[prev, nxt]

            if cost < best_cost:
                best_cost = cost
                best_insertion = (r_idx, pos)

    if best_insertion is not None:
        r, p = best_insertion
        ls.routes[r].insert(p, node)
        log.append((node, r, p))
        return True

    # No direct insertion possible - try ejection chain
    for r_idx, route in enumerate(ls.routes):
        if r_idx == excluded_route or not route:
            continue

        # Try ejecting a node from this route
        for eject_pos, eject_node in enumerate(route):
            eject_demand = ls.demands.get(eject_node, 0)
            load = ls._calc_load_fresh(route)

            # Check if we can fit the new node after ejection
            if load - eject_demand + node_demand > ls.Q:
                continue

            # Eject and try to reinsert recursively
            route.pop(eject_pos)

            # Insert new node
            best_pos = 0
            best_ins_cost = float("inf")
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0
                cost = ls.d[prev, node] + ls.d[node, nxt] - ls.d[prev, nxt]
                if cost < best_ins_cost:
                    best_ins_cost = cost
                    best_pos = pos

            route.insert(best_pos, node)
            log.append((node, r_idx, best_pos))

            # Now try to insert ejected node
            if _try_insert_with_chain(ls, eject_node, excluded_route, depth - 1, log):
                return True

            # Rollback
            route.pop(best_pos)
            route.insert(eject_pos, eject_node)
            log.pop()

    return False


def _rollback_ejections(
    ls: Any,
    log: List[Tuple[int, int, int]],
    source_route: int,
) -> None:
    """Rollback ejection chain insertions."""
    for node, r_idx, pos in reversed(log):
        if r_idx < len(ls.routes) and pos < len(ls.routes[r_idx]):
            if ls.routes[r_idx][pos] == node:
                ls.routes[r_idx].pop(pos)
                # Put back in source route
                ls.routes[source_route].append(node)
    log.clear()


def lambda_interchange(
    ls: Any,
    lambda_max: int = 2,
) -> bool:
    """
    λ-Interchange neighborhood search.

    Systematically explores cross-exchange moves with segments up to
    length λ. This is a wrapper that explores the full neighborhood.

    Args:
        ls: LocalSearch instance.
        lambda_max: Maximum segment length to exchange.

    Returns:
        True if any improving move was found.
    """
    improved = False

    for r_a in range(len(ls.routes)):
        for r_b in range(r_a + 1, len(ls.routes)):
            route_a = ls.routes[r_a]
            route_b = ls.routes[r_b]

            for seg_a_len in range(lambda_max + 1):
                for seg_b_len in range(lambda_max + 1):
                    if seg_a_len == 0 and seg_b_len == 0:
                        continue

                    for seg_a_start in range(max(1, len(route_a) - seg_a_len + 1)):
                        for seg_b_start in range(max(1, len(route_b) - seg_b_len + 1)):
                            if cross_exchange(
                                ls,
                                r_a,
                                seg_a_start,
                                seg_a_len,
                                r_b,
                                seg_b_start,
                                seg_b_len,
                            ):
                                improved = True
                                # Restart search after improvement
                                return True

    return improved
