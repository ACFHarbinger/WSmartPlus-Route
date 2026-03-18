"""
Local Branching (LB) matheuristic solver for VRPP.

Reference:
    Fischetti, M., & Lodi, A. (2003). "Local Branching".
    Mathematical Programming.
"""

from typing import Dict, List, Optional, Tuple

import gurobipy as gp
import numpy as np

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..kernel_search.solver import _dfj_subtour_elimination_callback, _reconstruct_tour, _set_mip_start, _setup_ks_model


def _add_local_branching_constraint(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    incumbent_x: Dict[Tuple[int, int], float],
    incumbent_y: Dict[int, float],
    k: int,
) -> gp.Constr:
    """
    Add the Local Branching constraint (Hamming distance <= k) to the model.

    The constraint is defined as:
    sum_{j in B0} x_j + sum_{j in B1} (1 - x_j) <= k
    where B0 is the set of binary variables that are 0 in the incumbent,
    and B1 is the set of binary variables that are 1 in the incumbent.
    """
    lhs = gp.LinExpr()

    # Process edge variables
    for key, var in x.items():
        val = incumbent_x.get(key, 0.0)
        if val < 0.5:
            lhs.addTerms(1.0, var)
        else:
            lhs.addConstant(1.0)
            lhs.addTerms(-1.0, var)

    # Process node selection variables
    for key, var in y.items():
        val = incumbent_y.get(key, 0.0)
        if val < 0.5:
            lhs.addTerms(1.0, var)
        else:
            lhs.addConstant(1.0)
            lhs.addTerms(-1.0, var)

    return model.addConstr(lhs <= k, name="local_branching_cut")


def run_local_branching_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    k: int = 10,
    max_iterations: int = 20,
    time_limit: float = 300.0,
    time_limit_per_iteration: float = 30.0,
    mip_limit_nodes: int = 5000,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve VRPP using Local Branching (LB).
    """
    model = gp.Model("LB_VRPP", env=env) if env else gp.Model("LB_VRPP")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # 1. Setup the core mathematical formulation with DFJ lazy constraints
    # CRITICAL: use_binary_vars=True because LB needs integer solutions, not LP relaxation
    x, y = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes, use_binary_vars=True)

    # 1.5. Warm-start with greedy heuristic
    # This is CRITICAL for Local Branching: we need an incumbent to define the k-neighborhood
    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # 2. Find Initial Solution
    # We solve the full MIP for a short time to get an initial incumbent.
    # We allocate up to 30% of the budget (or at least 20s if budget allows) for this critical step
    initial_alloc = min(time_limit * 0.8, max(20.0, time_limit * 0.3))
    model.setParam("TimeLimit", initial_alloc)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        # Fallback to a longer solve if no solution found in initial time
        fallback_alloc = min(time_limit * 0.8, max(15.0, time_limit * 0.5))
        model.setParam("TimeLimit", fallback_alloc)
        model.optimize(_dfj_subtour_elimination_callback)
        if model.SolCount == 0:
            return [0, 0], 0.0, 0.0

    current_best_obj = model.ObjVal
    incumbent_x = {key: var.X for key, var in x.items()}
    incumbent_y = {key: var.X for key, var in y.items()}

    # 3. Local Branching Loop
    start_time = model.Runtime
    iterations = 0
    lb_cut = None

    while iterations < max_iterations:
        elapsed = model.Runtime - start_time
        if elapsed > time_limit:
            break

        # Remove previous LB cut if it exists
        if lb_cut:
            model.remove(lb_cut)

        # Add new LB cut relative to current incumbent
        lb_cut = _add_local_branching_constraint(model, x, y, incumbent_x, incumbent_y, k)

        # Solve restricted sub-MIP with DFJ callback
        remaining_time = time_limit - elapsed
        iter_time = min(time_limit_per_iteration, remaining_time)
        model.setParam("TimeLimit", iter_time)
        model.setParam("NodeLimit", mip_limit_nodes)

        model.optimize(_dfj_subtour_elimination_callback)

        if model.SolCount > 0 and model.ObjVal > current_best_obj + 1e-4:
            # Improvement found! Update incumbent and move to new neighborhood
            current_best_obj = model.ObjVal
            incumbent_x = {key: var.X for key, var in x.items()}
            incumbent_y = {key: var.X for key, var in y.items()}
            # Option: could dynamically adjust k here (e.g., reset if it was increased)
        else:
            # No improvement found in the k-neighborhood.
            # Diversification: in simple LB, we might just stop or try a different constraint.
            # Here we follow the basic scheme: try to move to the other side of the cut if possible,
            # but for our simple implementation, we'll just stop if no progress is made.
            break

        iterations += 1

    # Final solve on the best-known neighborhood (or full model with best values)
    if lb_cut:
        model.remove(lb_cut)

    # Just ensure we have the best solution loaded
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)

    if recorder:
        recorder.record(engine="local_branching", obj_val=model.ObjVal, cost=cost, solved=1)

    return tour, float(model.ObjVal), float(cost)
