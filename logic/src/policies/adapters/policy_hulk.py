"""
Policy adapter for HULK hyper-heuristic.

Provides the interface between the simulator and the HULK solver.
"""

from typing import List

from ..hyper_heuristic_us_lk import HULKParams, HULKSolver


def policy_hulk(context) -> List[List[int]]:
    """
    HULK hyper-heuristic policy adapter.

    Args:
        context: Problem context containing:
            - dist_matrix: Distance matrix
            - wastes: Waste dictionary
            - capacity: Vehicle capacity
            - R: Revenue multiplier
            - C: Cost multiplier
            - params: HULK parameters (optional)
            - mandatory_nodes: Must-visit nodes (optional)
            - seed: Random seed (optional)

    Returns:
        Best routes found by HULK.
    """
    # Extract parameters from context
    params = context.params if hasattr(context, "params") and context.params is not None else HULKParams()

    # Create solver
    solver = HULKSolver(
        dist_matrix=context.dist_matrix,
        wastes=context.wastes,
        capacity=context.capacity,
        R=context.R,
        C=context.C,
        params=params,
        mandatory_nodes=getattr(context, "mandatory_nodes", None),
        seed=getattr(context, "seed", None),
    )

    # Solve
    best_routes, best_profit, best_cost = solver.solve()

    # Store results in context if possible
    if hasattr(context, "results"):
        context.results["profit"] = best_profit
        context.results["cost"] = best_cost

    return best_routes
