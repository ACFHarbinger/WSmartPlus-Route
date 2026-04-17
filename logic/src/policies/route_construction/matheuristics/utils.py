"""
Shared utilities for multi-period routing policies.

Thin wrappers around the canonical operator implementations in
logic.src.policies.helpers.operators.*. Do NOT re-implement operators here.
"""

from __future__ import annotations

import random
from typing import List, Optional

import numpy as np

# ---------------------------------------------------------------------------
# Problem-level helpers (these belong here, not in the operators directory,
# because they depend on ProblemContext)
# ---------------------------------------------------------------------------
from logic.src.interfaces.context.problem_context import ProblemContext

# ---------------------------------------------------------------------------
# Perturbation operators (canonical sources)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Repair operators
# ---------------------------------------------------------------------------
from logic.src.policies.helpers.operators.recreate_repair.farthest import (
    farthest_insertion,
    farthest_profit_insertion,
)

# ---------------------------------------------------------------------------
# Improvement operators (canonical sources)
# ---------------------------------------------------------------------------
from logic.src.policies.helpers.operators.solution_initialization import (
    build_grasp_routes,
    build_greedy_routes,
    build_nn_routes,
    build_regret_routes,
    build_savings_routes,
)


def initial_plan_greedy(
    problem: ProblemContext,
    rng: Optional[np.random.Generator] = None,
    method: str = "greedy",
) -> List[List[int]]:
    """
    Build an initial D-day plan using the specified initialization method.

    Advances the problem state day-by-day using mean fill increments.
    Returns a full D-day plan as List[List[int]] (one flat route per day,
    depot excluded).

    Args:
        problem:  ProblemContext for day 0.
        rng:      NumPy Generator for stochastic initialization methods.
        method:   One of "greedy", "nn", "savings", "regret", "grasp",
                  "farthest", "farthest_profit".

    The per-day route is extracted as the flattened union of all vehicle
    routes returned by the underlying initializer (single-vehicle assumption).
    """
    _rng_stdlib = random.Random(int(rng.integers(0, 2**31)) if rng is not None else 42)
    _np_rng = rng or np.random.default_rng(42)

    plan: List[List[int]] = []
    current = problem

    for _ in range(problem.horizon):
        routes = _build_single_day_routes(current, _rng_stdlib, _np_rng, method)
        flat = [node for route in routes for node in route]
        plan.append(flat)
        current = current.advance(flat)

    return plan


def _build_single_day_routes(
    problem: ProblemContext,
    rng_stdlib: random.Random,
    rng_np: np.random.Generator,
    method: str,
) -> List[List[int]]:
    """Dispatch single-day route construction to the appropriate operator."""
    dm = problem.distance_matrix
    wastes = problem.wastes
    Q = problem.capacity
    R = problem.revenue_per_kg
    C = problem.cost_per_km
    mandatory = problem.mandatory
    n = len(dm) - 1

    if method == "greedy":
        return build_greedy_routes(dm, wastes, Q, R, C, mandatory, rng=rng_stdlib)

    elif method == "nn":
        return build_nn_routes(
            nodes=list(range(1, n + 1)),
            mandatory_nodes=mandatory,
            wastes=wastes,
            capacity=Q,
            dist_matrix=dm,
            R=R,
            C=C,
            rng=rng_stdlib,
        )

    elif method == "savings":
        return build_savings_routes(dm, wastes, Q, R, C, mandatory)

    elif method == "regret":
        return build_regret_routes(dm, wastes, Q, R, C, mandatory, rng=rng_stdlib)

    elif method == "grasp":
        alpha = float(rng_np.uniform(0.0, 0.3))  # randomised greedy parameter
        return build_grasp_routes(dm, wastes, Q, R, C, mandatory, alpha=alpha, rng=rng_stdlib)

    elif method == "farthest":
        routes: List[List[int]] = [[m] for m in mandatory] if mandatory else [[]]
        removed = [i for i in range(1, n + 1) if i not in set(mandatory)]
        return farthest_insertion(routes, removed, dm, wastes, Q, mandatory_nodes=mandatory)

    elif method == "farthest_profit":
        routes = [[m] for m in mandatory] if mandatory else [[]]
        removed = [i for i in range(1, n + 1) if i not in set(mandatory)]
        return farthest_profit_insertion(routes, removed, dm, wastes, Q, R, C, mandatory_nodes=mandatory)

    else:
        raise ValueError(f"Unknown initialization method: {method!r}")


def route_profit(route: List[int], problem: ProblemContext) -> float:
    """Compute P(A_d, w_d) = r_w·Σw_i − c_km·dist for a single route."""
    revenue = sum(problem.wastes.get(i, 0.0) * problem.revenue_per_kg for i in route)
    path = [0] + route + [0]
    dist = sum(problem.distance_matrix[path[k]][path[k + 1]] for k in range(len(path) - 1))
    return revenue - problem.cost_per_km * dist


def route_cost(route: List[int], problem: ProblemContext) -> float:
    """Compute c_km·dist for a single route."""
    path = [0] + route + [0]
    return problem.cost_per_km * sum(problem.distance_matrix[path[k]][path[k + 1]] for k in range(len(path) - 1))
