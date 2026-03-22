"""
Global wrapper for Unstringing operators as a general destroy/removal heuristic.

Provides `unstringing_removal` and `unstringing_profit_removal` wrappers
that remove nodes by employing GENIUS Unstringing moves to gracefully extract
nodes from existing routes while rewiring them to minimize cost bumps.
"""

import random
from typing import Dict, List, Optional, Tuple

import numpy as np

from logic.src.policies.other.operators.unstringing_stringing.unstringing_i import (
    apply_type_i_us,
    apply_type_i_us_profit,
)
from logic.src.policies.other.operators.unstringing_stringing.unstringing_ii import (
    apply_type_ii_us,
    apply_type_ii_us_profit,
)
from logic.src.policies.other.operators.unstringing_stringing.unstringing_iii import (
    apply_type_iii_us,
    apply_type_iii_us_profit,
)
from logic.src.policies.other.operators.unstringing_stringing.unstringing_iv import (
    apply_type_iv_us,
    apply_type_iv_us_profit,
)


def _evaluate_routes(routes: List[List[int]], dist_matrix: np.ndarray) -> float:
    """Evaluate total route cost (negative for maximization)."""
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


def _apply_unstring_op(
    route: List[int],
    unstring_type: int,
    params: Tuple,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    profit_mode: bool,
) -> Tuple[List[int], float]:
    """Helper to apply unstringing operation and get new route and immediate profit val."""
    if unstring_type in (2, 4):
        i, j, k, l = params
        if profit_mode:
            if unstring_type == 2:
                return apply_type_ii_us_profit(route, i, j, k, l, dist_matrix, wastes, R, C)
            return apply_type_iv_us_profit(route, i, j, k, l, dist_matrix, wastes, R, C)
        new_route = apply_type_ii_us(route, i, j, k, l) if unstring_type == 2 else apply_type_iv_us(route, i, j, k, l)
        return new_route, 0.0

    i, j, k = params
    if profit_mode:
        if unstring_type == 1:
            return apply_type_i_us_profit(route, i, j, k, dist_matrix, wastes, R, C)
        return apply_type_iii_us_profit(route, i, j, k, dist_matrix, wastes, R, C)
    new_route = apply_type_i_us(route, i, j, k) if unstring_type == 1 else apply_type_iii_us(route, i, j, k)
    return new_route, 0.0


def _try_unstring_removal(
    routes: List[List[int]],
    r_idx: int,
    n_idx: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float,
    C: float,
    rng: random.Random,
    profit_mode: bool = False,
) -> Optional[Tuple[List[List[int]], float]]:
    """Try removing a node using the specified unstringing operator, returning the best result."""
    route = routes[r_idx]
    if len(route) < 3:
        return None

    best_routes = None
    best_value = -float("inf")

    i = n_idx
    target_node = route[i]

    for _ in range(min(5, len(route))):
        valid_positions = [p for p in range(len(route)) if p != i]

        try:
            if unstring_type in (1, 3):
                if len(valid_positions) < 2:
                    continue
                j = rng.choice(valid_positions)
                valid_k = [p for p in valid_positions if p != j]
                k = rng.choice(valid_k)
                params: Tuple[int, ...] = (i, j, k)
            else:
                if len(valid_positions) < 3:
                    continue
                j = rng.choice(valid_positions)
                valid_k = [p for p in valid_positions if p != j]
                k = rng.choice(valid_k)
                valid_l = [p for p in valid_positions if p not in (j, k)]
                l = rng.choice(valid_l)
                params = (i, j, k, l)  # type: ignore[assignment]

            new_route, val = _apply_unstring_op(route, unstring_type, params, dist_matrix, wastes, R, C, profit_mode)

            # Validate removal
            if target_node in new_route:
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


def unstringing_removal_wrapper(
    routes: List[List[int]],
    n_remove: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 0.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
    profit_mode: bool = False,
) -> Tuple[List[List[int]], List[int]]:
    """
    Core unstringing removal logic.
    Fallback to worst removal if unstringing fails (e.g., route too short).
    """
    if rng is None:
        rng = random.Random(42)

    removed_nodes = []
    for _ in range(n_remove):
        best_routes = None
        best_value = -float("inf")
        best_node = -1

        # Consider all currently assigned nodes
        for r_idx, route in enumerate(routes):
            if len(route) < 3:
                continue
            for n_idx, node in enumerate(route):
                result = _try_unstring_removal(
                    routes, r_idx, n_idx, unstring_type, dist_matrix, wastes, R, C, rng, profit_mode
                )
                if result:
                    test_routes, val = result
                    if val > best_value:
                        best_value = val
                        best_routes = test_routes
                        best_node = node

        if best_node != -1 and best_routes is not None:
            routes = best_routes
            removed_nodes.append(best_node)
            routes = [r for r in routes if r]
        else:
            # Fallback for remaining nodes that couldn't be unstrung
            from logic.src.policies.other.operators.destroy.worst import (  # noqa: PLC0415
                worst_profit_removal,
                worst_removal,
            )

            nodes_to_remove = min(1, sum(len(r) for r in routes))
            if nodes_to_remove == 0:
                break

            if profit_mode:
                routes, rem = worst_profit_removal(
                    routes,
                    nodes_to_remove,
                    dist_matrix,
                    wastes,
                    R,
                    C,
                )
            else:
                routes, rem = worst_removal(
                    routes,
                    nodes_to_remove,
                    dist_matrix,
                )
            removed_nodes.extend(rem)

    return routes, removed_nodes


# ==== Public Wrapper API ====


def unstringing_removal(
    routes: List[List[int]],
    n_remove: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    rng: Optional[random.Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """Standard CVRP Unstringing Removal."""
    return unstringing_removal_wrapper(
        routes=routes,
        n_remove=n_remove,
        unstring_type=unstring_type,
        dist_matrix=dist_matrix,
        wastes={},
        rng=rng,
        profit_mode=False,
    )


def unstringing_profit_removal(
    routes: List[List[int]],
    n_remove: int,
    unstring_type: int,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Tuple[List[List[int]], List[int]]:
    """VRPP Cost-Aware Unstringing Removal."""
    # Feedback implementation: Keep Variants I, II, and III purely structural
    profit_mode = unstring_type not in (1, 2, 3)
    return unstringing_removal_wrapper(
        routes=routes,
        n_remove=n_remove,
        unstring_type=unstring_type,
        dist_matrix=dist_matrix,
        wastes=wastes,
        R=R,
        C=C,
        rng=rng,
        profit_mode=profit_mode,
    )
