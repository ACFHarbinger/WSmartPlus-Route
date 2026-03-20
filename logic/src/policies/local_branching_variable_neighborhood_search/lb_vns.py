"""
Local Branching with Variable Neighborhood Search (LB-VNS) matheuristic solver.

This implementation follows the hybrid approach proposed by Hansen et al. (2006),
which integrates the metaheuristic structure of Variable Neighborhood Search (VNS)
with the local search capabilities of Local Branching (LB).

Algorithm Overview:
    1. Initialization: Find an initial feasible solution (incumbent).
    2. VNS Cycle:
        a. Shaking: Randomly perturb the incumbent to find a starting point
           in a neighborhood N_k (defined by Hamming distance k).
        b. Local Search: Apply Local Branching from the shaken solution
           to find a local optimum or improved solution.
        c. Move/Stay: If improvement is found, update incumbent and reset k.
           Otherwise, increase k (neighborhood expansion).

Reference:
    Hansen, P., Mladenović, N., & Urošević, D. (2006).
    "Variable neighborhood search and local branching".
    Computers & Operations Research.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    gp = None
    GRB = None

from ..kernel_search.solver import _dfj_subtour_elimination_callback, _reconstruct_tour, _set_mip_start, _setup_ks_model
from ..local_branching.lb import _add_local_branching_constraint


def _shake_solution_gurobi(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    incumbent_x: Dict[Tuple[int, int], float],
    incumbent_y: Dict[int, float],
    k: int,
    seed: int = 42,
) -> Tuple[Optional[Dict[Tuple[int, int], float]], Optional[Dict[int, float]]]:
    """
    Perform the Shaking phase by finding a 'random' solution in the k-neighborhood.

    In the context of MILP, shaking is implemented by adding a Hamming distance
    constraint (dist <= k) and a lower bound (dist >= k-delta) to ensure the
    solution moves 'away' from the incumbent, then solving with a random objective.

    This ensures that the starting point for the subsequent Local Search is
    sufficiently diversified while remaining within the vicinity of the
    promising search space.

    Args:
        model (gp.Model): The pre-configured Gurobi model (VRPP formulation).
        x (Dict[Tuple[int, int], gp.Var]): Dictionary of binary edge variables.
        y (Dict[int, gp.Var]): Dictionary of binary node selection variables.
        incumbent_x (Dict[Tuple[int, int], float]): Values of x in the current best solution.
        incumbent_y (Dict[int, float]): Values of y in the current best solution.
        k (int): Hamming distance radius for the current VNS neighborhood.
        seed (int): Random seed for generating the stochastic objective.

    Returns:
        Tuple[Optional[Dict[Tuple[int, int], float]], Optional[Dict[int, float]]]:
            - shaken_x: Values of edge variables in the new starting solution.
            - shaken_y: Values of node variables in the new starting solution.
            Returns (None, None) if the k-neighborhood is infeasible.
    """
    # 1. Save original objective and parameters to restore them later
    orig_obj = model.getObjective()
    orig_sol_limit = model.getParamInfo("SolutionLimit")[2]
    orig_time_limit = model.getParamInfo("TimeLimit")[2]

    # 2. Add Hamming distance constraints to define the 'shell' of the k-neighborhood
    # Hamming distance H(s, s') = sum(|x_i - x_i'|)
    lhs = gp.LinExpr()
    for key, var in x.items():
        val = incumbent_x.get(key, 0.0)
        if val < 0.5:
            lhs.addTerms(1.0, var)  # (1 - 0) * var = var
        else:
            lhs.addConstant(1.0)
            lhs.addTerms(-1.0, var)  # (1 - 1) * var is not right.
            # Local Branching formula: sum_{j in S} (1 - x_j) + sum_{j not in S} x_j <= k
            # where S is the set of indices where the incumbent has value 1.
    for key, var in y.items():
        val = incumbent_y.get(key, 0.0)
        if val < 0.5:
            lhs.addTerms(1.0, var)
        else:
            lhs.addConstant(1.0)
            lhs.addTerms(-1.0, var)

    # Restrict search to the boundary of the k-neighborhood to force diversification
    shake_cut_upper = model.addConstr(lhs <= k, name="shake_cut_upper")
    shake_cut_lower = model.addConstr(lhs >= max(1, k - 2), name="shake_cut_lower")

    # 3. Set a purely random objective to promote unbiased diversification
    # This prevents the solver from just finding the 'next best' solution,
    # encouraging exploration of the feasible region.
    rng = np.random.default_rng(seed)
    random_obj = gp.LinExpr()
    for var in model.getVars():
        if var.VType in [GRB.BINARY, GRB.INTEGER]:
            random_obj.addTerms(rng.standard_normal(), var)

    model.setObjective(random_obj, GRB.MINIMIZE)

    # 4. Find the first feasible solution as quickly as possible
    model.setParam("SolutionLimit", 1)  # Stop at first feasible
    model.setParam("TimeLimit", 5.0)  # Don't spend too much time shaking
    model.optimize()

    res_x, res_y = None, None
    if model.SolCount > 0:
        res_x = {key: var.X for key, var in x.items()}
        res_y = {key: var.X for key, var in y.items()}

    # 5. Restore original model state (cleanup)
    model.remove(shake_cut_upper)
    model.remove(shake_cut_lower)
    model.setObjective(orig_obj)
    model.setParam("SolutionLimit", orig_sol_limit)
    model.setParam("TimeLimit", orig_time_limit)

    return res_x, res_y


def run_lb_vns_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    k_min: int = 10,
    k_max: int = 50,
    k_step: int = 5,
    time_limit: float = 300.0,
    time_limit_per_lb: float = 30.0,
    max_lb_iterations: int = 10,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve the Vehicle Routing Problem with Profits (VRPP) using LB-VNS.

    LB-VNS iterates through neighborhoods of increasing size (Hamming distance).
    In each iteration, it 'shakes' the incumbent to a new point and then
    intensifies the search using Local Branching.

    Mathematical Formulation:
        The underlying model is a Prize-Collecting VRP / VRP with Profits,
        implemented using MTZ (Miller-Tucker-Zemlin) subtour elimination constraints.

    Args:
        dist_matrix (np.ndarray): Symmetric distance matrix between all nodes.
        wastes (Dict[int, float]): Dictionary mapping node indices to profits.
        capacity (float): Maximum vehicle load.
        R (float): Revenue multiplier for node collection.
        C (float): Cost multiplier for distance traveled.
        mandatory_nodes (List[int]): Nodes that MUST be included in the tour.
        k_min (int): Minimum Hamming distance for early search stages.
        k_max (int): Maximum Hamming distance (exploration ceiling).
        k_step (int): Increment for 'k' when no improvement is found.
        time_limit (float): Overall wall-clock budget for the search.
        time_limit_per_lb (float): Time budget for each Local Branching call.
        max_lb_iterations (int): Loop limit for internal Local Branching.
        mip_gap (float): Optimality gap for sub-problem termination.
        seed (int): Global seed for randomization.
        env (Optional[gp.Env]): Shared Gurobi environment to reduce startup overhead.

    Returns:
        Tuple[List[int], float, float]:
            - tour: The node sequence representing the best found tour.
            - objective_value: The final mathematical objective score.
            - total_distance: The physical distance cost of the tour.
    """
    # Initialize the Gurobi model
    model = gp.Model("LB_VNS_VRPP", env=env) if env else gp.Model("LB_VRPP")
    model.setParam("OutputFlag", 0)  # Silent mode
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # 1. Setup mathematical formulation with DFJ lazy constraints (no MTZ)
    # This creates the variables x (edges) and y (nodes) and the base constraints.
    # CRITICAL: use_binary_vars=True because LB-VNS needs integer solutions, not LP relaxation
    x, y = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes, use_binary_vars=True)

    # 1.5. Warm-start with greedy heuristic
    # This is CRITICAL for LB-VNS: we need an incumbent to define the k-neighborhood
    _set_mip_start(model, x, y, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # 2. Find an initial feasible solution
    # A short initial B&B run to establish a baseline (incumbent).
    # We allocate up to 30% of the budget (or at least 20s if budget allows) for this critical step
    initial_alloc = min(time_limit * 0.8, max(20.0, time_limit * 0.3))
    model.setParam("TimeLimit", initial_alloc)
    model.optimize(_dfj_subtour_elimination_callback)

    if model.SolCount == 0:
        # If no solution found, try a slightly longer emergency solve.
        fallback_alloc = min(time_limit * 0.8, max(15.0, time_limit * 0.5))
        model.setParam("TimeLimit", fallback_alloc)
        model.optimize(_dfj_subtour_elimination_callback)
        if model.SolCount == 0:
            return [0, 0], 0.0, 0.0

    current_best_obj = model.ObjVal
    incumbent_x = {key: var.X for key, var in x.items()}
    incumbent_y = {key: var.X for key, var in y.items()}

    # 3. MAIN VARIABLE NEIGHBORHOOD SEARCH LOOP
    # =========================================================================
    start_time = time.process_time()
    k = k_min

    while k <= k_max:
        elapsed = time.process_time() - start_time
        if elapsed > time_limit:
            # Terminate if the global budget is exhausted.
            break

        # --- PHASE 1: SHAKING (Exploration) ---
        # Find a diversified starting point in the k-neighborhood.
        shaken_x, shaken_y = _shake_solution_gurobi(model, x, y, incumbent_x, incumbent_y, k, seed + k)

        if shaken_x is None:
            # If the neighborhood is infeasible (e.g., k is too small to flip mandatory edges),
            # expand the neighborhood and try again.
            k += k_step
            continue

        # --- PHASE 2: LOCAL SEARCH (Intensification) ---
        # Apply Local Branching to improve the shaken solution.
        # k_ls is a smaller, fixed radius (10) for focused intensification.
        k_ls = 10
        ls_cut = _add_local_branching_constraint(model, x, y, shaken_x, shaken_y, k_ls)

        # Warm-start the solver with the 'shaken' solution.
        for key, var in x.items():
            var.Start = shaken_x.get(key, 0.0)
        for key, var in y.items():
            var.Start = shaken_y.get(key, 0.0)

        remaining_time = time_limit - (time.process_time() - start_time)
        iter_time = min(time_limit_per_lb, remaining_time)
        model.setParam("TimeLimit", iter_time)
        model.optimize(_dfj_subtour_elimination_callback)

        # --- PHASE 3: NEIGHBORHOOD CHANGE (Move Acceptance) ---
        # Check if the intensification phase yielded a solution better than the GLOBAL incumbent.
        if model.SolCount > 0 and model.ObjVal > current_best_obj + 1e-4:
            # Moving: A better attraction basin was found! Update and intensification restarts.
            current_best_obj = model.ObjVal
            incumbent_x = {key: var.X for key, var in x.items()}
            incumbent_y = {key: var.X for key, var in y.items()}
            k = k_min  # Reset VNS to smallest neighborhood
        else:
            # Staying: Neighborhood N_k exhausted. Expand to diversify further.
            k += k_step

        # Cleanup the Local Branching cut for the next iteration.
        model.remove(ls_cut)
        for var in model.getVars():
            var.Start = GRB.UNDEFINED

    # 4. FINAL SOLUTION EXTRACTION
    # =========================================================================
    # After the loop terminates, reconstruct the best found tour sequence.
    tour, cost = _reconstruct_tour(len(dist_matrix), incumbent_x, dist_matrix)

    return tour, float(current_best_obj), float(cost)
