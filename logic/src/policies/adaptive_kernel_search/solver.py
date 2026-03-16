"""
Adaptive Kernel Search (AKS) matheuristic solver for VRPP.

Reference:
    Guastaroba, G., Savelsbergh, M., & Speranza, M. G. (2017). "Adaptive Kernel
    Search: A heuristic for solving Mixed Integer linear Programs".
    European Journal of Operational Research.
"""

from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from logic.src.tracking.viz_mixin import PolicyStateRecorder

from ..kernel_search.solver import _reconstruct_tour, _setup_ks_model


def _get_partitioned_vars_aks(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    initial_kernel_size: int,
    bucket_size: int,
) -> Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]:
    """
    Solve the LP relaxation of the VRPP model and partition variables into Kernel and Buckets.

    This selection phase identifies the most promising decision variables (edges and nodes)
    to form the initial optimization 'Kernel'. The ranking is based on the fractional
    values of the variables in the optimal solution to the continuous relaxation.

    Args:
        model (gp.Model): The mathematical programming model.
        x (Dict[Tuple[int, int], gp.Var]): Map of edge index pairs to Gurobi binary variables.
        y (Dict[int, gp.Var]): Map of customer IDs to Gurobi binary selection variables.
        initial_kernel_size (int): Targeted number of variables for the starting Kernel.
        bucket_size (int): Targeted number of variables per initial partition bucket.

    Returns:
        Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]:
            - kernel_vars: The subset of variables selected for the initial math model.
            - remaining_vars: Sorted list of all variables not in the kernel.
            - buckets: Initial grouping of `remaining_vars` into disjoint sets.
    """
    # 1. Solve the continuous (LP) relaxation
    model.optimize()
    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or model.SolCount == 0:
        return [], [], []

    # 2. Extract and sort variables by their fractional values (nearness to 1.0)
    all_vars = []
    for var in x.values():
        all_vars.append((var, var.X))
    for var in y.values():
        all_vars.append((var, var.X))

    # Sort descending: Variables with value 1.0 are most likely to be in the integer optimum
    all_vars.sort(key=lambda item: item[1], reverse=True)

    # 3. Restore variable types to BINARY for the subsequent MIP stages
    for v_obj, _ in all_vars:
        v_obj.vtype = GRB.BINARY

    # 4. Partition variables based on the sorted ranking
    kernel_vars = [v for v, _ in all_vars[:initial_kernel_size]]
    remaining_vars = [v for v, _ in all_vars[initial_kernel_size:]]

    # Organize remaining variables into sequential buckets for the iterative improvement phase
    buckets = [remaining_vars[i : i + bucket_size] for i in range(0, len(remaining_vars), bucket_size)]

    return kernel_vars, remaining_vars, buckets


def _solve_aks_iterations(
    model: gp.Model,
    kernel_vars: List[gp.Var],
    remaining_vars: List[gp.Var],
    initial_bucket_size: int,
    max_buckets: int,
    bucket_growth_factor: float,
    time_limit: float,
    mip_limit_nodes: int,
) -> Set[gp.Var]:
    """
    Execute the core Adaptive Kernel Search iterative improvement loop.

    This function dynamically manages the search by solving restricted sub-MIPs.
    It implements the paper's adaptive logic: variable promotion and dynamic
    bucket sizing based on solution progress.

    Args:
        model (gp.Model): The VRPP mathematical model.
        kernel_vars (List[gp.Var]): Variables currently active in the optimization search.
        remaining_vars (List[gp.Var]): Variables currently fixed to 0.
        initial_bucket_size (int): Starting number of variables added in each iteration.
        max_buckets (int): Limit on total improvement iterations.
        bucket_growth_factor (float): Multiplier for bucket expansion when improvements occur.
        time_limit (float): Total wall-clock budget for the improvement phase.
        mip_limit_nodes (int): Hard node limit for each sub-MIP solve.

    Returns:
        Set[gp.Var]: The set of "useful" variables (those used in the best solutions)
            that should be included in the final high-quality solve.
    """
    # Initially fix ALL non-kernel variables to zero
    for v in remaining_vars:
        v.ub = 0

    best_obj = -float("inf")
    used_vars = set()
    start_time = model.Runtime

    # PHASE 1: Initial Kernel Solve
    # Find the best possible solution using only nodes/edges in the initial kernel.
    model.setParam("TimeLimit", max(5.0, time_limit * 0.1))
    model.setParam("NodeLimit", mip_limit_nodes)
    model.optimize()

    if model.SolCount > 0:
        best_obj = model.ObjVal
        used_vars = {v for v in kernel_vars if v.X > 0.5}

    # PHASE 2: Adaptive Iterative Refinement
    current_idx = 0
    current_bucket_size = initial_bucket_size
    buckets_solved = 0

    while current_idx < len(remaining_vars) and buckets_solved < max_buckets:
        elapsed = model.Runtime - start_time
        if elapsed > time_limit:
            break

        # Select the 'slice' of remaining variables for this bucket
        end_idx = min(current_idx + current_bucket_size, len(remaining_vars))
        current_bucket = remaining_vars[current_idx:end_idx]

        # Temporarily 'unfix' variables in the current bucket
        for v in current_bucket:
            v.ub = 1

        # Adaptive Time Allocation:
        # Calculate time for this sub-MIP based on total budget and remaining tasks.
        remaining_time = time_limit - elapsed
        remaining_buckets = max_buckets - buckets_solved
        iter_time = max(2.0, remaining_time / max(1, remaining_buckets))
        model.setParam("TimeLimit", iter_time)

        model.optimize()

        improved = False
        if model.SolCount > 0:
            if model.ObjVal > best_obj:
                # SUCCESS: Found a better solution by integrating bucket variables
                best_obj = model.ObjVal
                improved = True

                # VARIABLE PROMOTION:
                # Permanently add variables that took positive values to the 'used set'
                for v in current_bucket:
                    if v.X > 0.5:
                        used_vars.add(v)

            # Fix variables that were NOT used (even if improved) back to 0 to keep model lean
            for v in current_bucket:
                if v.X < 0.5:
                    v.ub = 0
        else:
            # Revert all variables in the bucket if no feasible solution was found
            for v in current_bucket:
                v.ub = 0

        # ADAPTIVE STEP: Adjustment of search intensity
        if improved:
            # If we found an improvement, we grow the bucket size for the next step.
            # Rationale: Success in one area suggests synergies nearby that might
            # require a larger simultaneous search to unlock.
            current_bucket_size = int(current_bucket_size * bucket_growth_factor)
        else:
            # If no improvement, continue with current size to manage complexity
            pass

        current_idx = end_idx
        buckets_solved += 1

    return used_vars


def run_adaptive_kernel_search_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    initial_kernel_size: int = 50,
    bucket_size: int = 20,
    max_buckets: int = 15,
    bucket_growth_factor: float = 1.2,
    time_limit: float = 300.0,
    mip_limit_nodes: int = 10000,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Solve the Vehicle Routing Problem with Profits (VRPP) using Adaptive Kernel Search (AKS).

    This entry point orchestrates the mathematical modeling, variable partitioning,
    adaptive iterative solving, and result reconstruction.

    Args:
        dist_matrix (np.ndarray): Symmetric or asymmetric cost matrix for nodes.
        wastes (Dict[int, float]): Map of node index to current bin fill level.
        capacity (float): Maximum vehicle capacity.
        R (float): Revenue multiplier for the objective function.
        C (float): Cost/Distance multiplier for the objective function.
        mandatory_nodes (List[int]): Nodes that MUST be visited regardless of profit.
        initial_kernel_size (int): size of the starting variable pool.
        bucket_size (int): starting size for search buckets.
        max_buckets (int): limit on improvement attempts.
        bucket_growth_factor (float): rate of neighborhood expansion.
        time_limit (float): total time budget for optimization.
        mip_limit_nodes (int): node limit for internal Gurobi solves.
        mip_gap (float): acceptable relative optimality gap.
        seed (int): random seed for solver consistency.
        env (Optional[gp.Env]): Shared Gurobi environment (useful for license management).
        recorder (Optional[PolicyStateRecorder]): System-wide state tracking hook.

    Returns:
        Tuple[List[int], float, float]:
            - tour: The optimized sequence of node IDs.
            - obj_val: The final MILP objective value achieved.
            - cost: The total travel distance/cost of the final tour.
    """
    # 1. Initialize the Gurobi model and set global parameters
    model = gp.Model("AKS_VRPP", env=env) if env else gp.Model("AKS_VRPP")
    model.setParam("OutputFlag", 0)
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # 2. Setup the core mathematical formulation (Variables and Constraints)
    # This includes the VRPP profit model + MTZ/Miller-Tucker-Zemlin subtour elimination.
    x, y, _ = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # 3. Perform Selection Phase via LP Relaxation
    kernel_vars, remaining_vars, _ = _get_partitioned_vars_aks(model, x, y, initial_kernel_size, bucket_size)

    if not kernel_vars:
        return [0, 0], 0.0, 0.0

    # 4. Execute the Adaptive Improvement Loop
    # Identifies the 'useful' subset of all variables found across iterations.
    used_vars = _solve_aks_iterations(
        model, kernel_vars, remaining_vars, bucket_size, max_buckets, bucket_growth_factor, time_limit, mip_limit_nodes
    )

    if not used_vars:
        return [0, 0], 0.0, 0.0

    # 5. Final Intensification
    # Solve one last MIP restricted to ONLY the variables that proved useful in
    # previous steps. This ensures the best combination is found and feasible.
    for v in used_vars:
        v.ub = 1
    model.optimize()

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    # 6. Tour Reconstruction
    # Extract the sequence of node visits from the binary edge variables (`x`).
    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)

    # 7. Telemetry / Logging
    if recorder:
        recorder.record(engine="adaptive_kernel_search", obj_val=model.ObjVal, cost=cost, solved=1)

    return tour, float(model.ObjVal), float(cost)
