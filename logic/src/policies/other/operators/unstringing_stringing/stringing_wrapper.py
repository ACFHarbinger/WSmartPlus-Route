"""
Global wrapper for Stringing operators as a general repair heuristic.

Provides `stringing_insertion` and `stringing_profit_insertion` wrappers
that mirror `greedy_insertion`, capable of iterating over unassigned nodes
(with `expand_pool` support) and inserting them using GENIUS Stringing moves.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.other.operators.unstringing_stringing.stringing_i import apply_type_i_s, apply_type_i_s_profit
from logic.src.policies.other.operators.unstringing_stringing.stringing_ii import (
    apply_type_ii_s,
    apply_type_ii_s_profit,
)
from logic.src.policies.other.operators.unstringing_stringing.stringing_iii import (
    apply_type_iii_s,
    apply_type_iii_s_profit,
)
from logic.src.policies.other.operators.unstringing_stringing.stringing_iv import (
    apply_type_iv_s,
    apply_type_iv_s_profit,
)


def _evaluate_routes(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """Evaluate total route cost."""
    cost = 0.0
    for rt in routes:
        if not rt:
            continue
        d = dist_matrix[0][rt[0]]
        for idx in range(len(rt) - 1):
            d += dist_matrix[rt[idx]][rt[idx + 1]]
        d += dist_matrix[rt[-1]][0]
        cost += d
    return -cost


def _apply_stringing_op(
    route: List[int],
    node: int,
    string_type: int,
    params: Tuple,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    profit_mode: bool,
) -> Tuple[List[int], float]:
    """Helper to apply stringing operation and get new route and immediate profit val."""
    if string_type in (2, 4):
        i, j, k, l = params
        if profit_mode:
            if string_type == 2:
                return apply_type_ii_s_profit(route, node, i, j, k, l, dist_matrix, wastes, capacity, R, C)
            return apply_type_iv_s_profit(route, node, i, j, k, l, dist_matrix, wastes, capacity, R, C)
        new_route = (
            apply_type_ii_s(route, node, i, j, k, l) if string_type == 2 else apply_type_iv_s(route, node, i, j, k, l)
        )
        return new_route, 0.0

    i, j, k = params
    if profit_mode:
        if string_type == 1:
            return apply_type_i_s_profit(route, node, i, j, k, dist_matrix, wastes, capacity, R, C)
        return apply_type_iii_s_profit(route, node, i, j, k, dist_matrix, wastes, capacity, R, C)
    new_route = apply_type_i_s(route, node, i, j, k) if string_type == 1 else apply_type_iii_s(route, node, i, j, k)
    return new_route, 0.0


def _try_string_insertion(
    routes: List[List[int]],
    node: int,
    r_idx: int,
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    rng: random.Random,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """Try inserting a node using the specified stringing operator, returning the best result."""
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = float("-inf")

    for _ in range(min(5, len(route))):
        valid_positions = list(range(len(route)))
        if len(valid_positions) < (4 if string_type in (2, 4) else 3):
            continue

        try:
            i = rng.choice(valid_positions)
            valid_j = [p for p in valid_positions if p != i]
            if not valid_j:
                continue
            j = rng.choice(valid_j)
            valid_k = [p for p in valid_positions if p not in (i, j)]
            if not valid_k:
                continue
            k = rng.choice(valid_k)

            if string_type in (2, 4):
                valid_l = [p for p in valid_positions if p not in (i, j, k)]
                if not valid_l:
                    continue
                l = rng.choice(valid_l)
                params: Tuple[int, ...] = (i, j, k, l)
            else:
                params = (i, j, k)  # type: ignore[assignment]

            new_route, val = _apply_stringing_op(
                route, node, string_type, params, dist_matrix, wastes, capacity, R, C, profit_mode
            )

            # Validate insertion
            if node not in new_route or sum(wastes.get(n, 0) for n in new_route) > capacity:
                continue

            test_routes = [list(r) for r in routes]
            test_routes[r_idx] = new_route

            if not profit_mode:
                val = _evaluate_routes(test_routes, dist_matrix)

            if val > best_value:
                best_value = val
                best_routes = test_routes

        except Exception:
            continue

    return (best_routes, best_value) if best_routes is not None else None


def stringing_insertion_wrapper(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float = 0.0,
    C: float = 1.0,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    profit_mode: bool = False,
    expand_pool: bool = False,
) -> List[List[int]]:
    """
    Core stringing insertion logic handling expanding pools and iteration.
    Fallback to greedy insertion if stringing fails (e.g., node cannot fit, route too short).
    """
    if rng is None:
        rng = random.Random(42)

    mandatory_nodes_set = set(mandatory_nodes) if mandatory_nodes else set()

    # Establish pool of unassigned candidates
    if expand_pool:
        visited = {n for r in routes for n in r}
        n_nodes = len(dist_matrix) - 1
        unassigned = sorted(list(set(range(1, n_nodes + 1)) - visited))
    else:
        unassigned = sorted(list(removed_nodes))

    # Iterate until pool is resolved
    while unassigned:
        best_routes = None
        best_value = -float("inf")
        best_node = -1

        # Try inserting every available unassigned node into every route
        for node in unassigned:
            is_mandatory = node in mandatory_nodes_set

            for r_idx in range(len(routes)):
                result = _try_string_insertion(
                    routes, node, r_idx, string_type, dist_matrix, wastes, capacity, R, C, rng, profit_mode
                )
                if result:
                    test_routes, val = result

                    if profit_mode:
                        # Bias score slightly for mandatory nodes in VRPP mode
                        effective_val = val + (1e9 if is_mandatory else 0)
                        # Reject negative pure-profit insertions unless mandatory
                        if not is_mandatory and val < -1e-4:
                            continue

                        if effective_val > best_value:
                            best_value = effective_val
                            best_routes = test_routes
                            best_node = node
                    else:
                        if val > best_value:
                            best_value = val
                            best_routes = test_routes
                            best_node = node

        # Apply best found insertion
        if best_node != -1 and best_routes is not None:
            routes = best_routes
            unassigned.remove(best_node)
        else:
            # Fallback for remaining nodes that couldn't be strung (too small routes, complex constraints)
            from logic.src.policies.other.operators.repair.greedy import (  # noqa: PLC0415
                greedy_insertion,
                greedy_profit_insertion,
            )

            if profit_mode:
                routes = greedy_profit_insertion(
                    routes,
                    unassigned,
                    dist_matrix,
                    wastes,
                    capacity,
                    R,
                    C,
                    mandatory_nodes,
                    expand_pool=False,  # Already isolated to unassigned pool
                )
            else:
                routes = greedy_insertion(
                    routes,
                    unassigned,
                    dist_matrix,
                    wastes,
                    capacity,
                    R=R if profit_mode else None,
                    mandatory_nodes=mandatory_nodes,
                    cost_unit=C,
                    expand_pool=False,
                )
            break  # Greedy fallback completely drains the unassigned array

    return routes


# ==== Public Wrapper API ====


def stringing_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """Standard CVRP Stringing Repair."""
    return stringing_insertion_wrapper(
        routes,
        removed_nodes,
        string_type,
        dist_matrix,
        wastes,
        capacity,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
        profit_mode=False,
        expand_pool=expand_pool,
    )


def stringing_profit_insertion(
    routes: List[List[int]],
    removed_nodes: List[int],
    string_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: Optional[List[int]] = None,
    rng: Optional[random.Random] = None,
    expand_pool: bool = False,
) -> List[List[int]]:
    """VRPP Cost-Aware Stringing Repair."""
    # Feedback implementation: Keep Variants I, II, and III purely structural
    profit_mode = string_type not in (1, 2, 3)
    return stringing_insertion_wrapper(
        routes,
        removed_nodes,
        string_type,
        dist_matrix,
        wastes,
        capacity,
        R,
        C,
        mandatory_nodes=mandatory_nodes,
        rng=rng,
        profit_mode=profit_mode,
        expand_pool=expand_pool,
    )
