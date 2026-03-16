"""
Kernel Search matheuristic solver for VRPP.

This module implements the core Gurobi-based Kernel Search algorithm.
The framework generalizes decomposition techniques by restricting binary variables
to a promising subset (the Kernel) and iteratively expanding it with additional sets (Buckets).

Reference:
    Angelelli, R., Dell'Amico, M., & Martello, S. (2010). "A Kernel Search Algorithm for the
    Vehicle Routing Problem with Pickups and Deliveries". Networks, 56(4), 297–307.
    Angelelli, R., & Mansini, R. (2010). "Kernel Search: a new heuristic framework for
    portfolio selection".
"""

from typing import Dict, List, Optional, Set, Tuple

import gurobipy as gp
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _setup_ks_model(
    model: gp.Model,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
) -> Tuple[Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var], Dict[int, gp.Var]]:
    """
    Setup the base mathematical parameters, variables, and constraints for the VRPP model.

    Args:
        model (gp.Model): The Gurobi model instance.
        dist_matrix (np.ndarray): Cost matrix between all nodes (0 is depot).
        wastes (Dict[int, float]): Waste level at each node.
        capacity (float): Capacity of the vehicle.
        R (float): Revenue multiplier for collected waste.
        C (float): Cost multiplier for distance traveled.
        mandatory_nodes (List[int]): Nodes that MUST have their `y` variable set to 1.

    Returns:
        Tuple: A 3-tuple containing:
            - `x` (Dict[Tuple[int, int], gp.Var]): binary/continuous flow variables x_{i,j}.
            - `y` (Dict[int, gp.Var]): binary/continuous selection variables y_i.
            - `u` (Dict[int, gp.Var]): continuous MTZ subtour elimination variables.
    """
    num_nodes = len(dist_matrix)
    nodes = list(range(num_nodes))
    customers = list(range(1, num_nodes))
    m_set = set(mandatory_nodes)

    # 1. Decision Variables
    # Flow variables: x[i,j] = 1 if the vehicle travels from i to j
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                # Initially continuous to allow LP relaxation sorting
                x[i, j] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    # Selection variables: y[i] = 1 if customer i is visited
    y = {}
    for i in customers:
        y[i] = model.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name=f"y_{i}")

    # Subtour elimination variables (MTZ formulation)
    u = {}
    for i in customers:
        # Load accumulated after visiting node i
        u[i] = model.addVar(lb=wastes.get(i, 0), ub=capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    # 2. Objective Function: Maximize (Revenue - Cost)
    travel_cost = quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)
    revenue = quicksum(wastes.get(i, 0) * R * y[i] for i in customers)
    model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

    # 3. Flow Balance and Selection Constraints
    for i in customers:
        # If node i is visited, exactly one edge enters and one edge leaves
        model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i])
        model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i])

    # Depot usage: at most one trip leaves the depot
    model.addConstr(quicksum(x[0, j] for j in customers) <= 1)
    # Conservation of flow at the depot
    model.addConstr(quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers))

    # 4. Miller-Tucker-Zemlin (MTZ) Subtour Elimination
    for i in customers:
        for j in customers:
            if i != j:
                dj = wastes.get(j, 0)
                # u[j] >= u[i] + dj - Q * (1 - x[i,j])
                model.addConstr(u[j] >= u[i] + dj - capacity * (1 - x[i, j]))

    # 5. Mandatory Visits
    for i in m_set:
        model.addConstr(y[i] == 1)

    return x, y, u


def _get_partitioned_vars(
    model: gp.Model,
    x: Dict[Tuple[int, int], gp.Var],
    y: Dict[int, gp.Var],
    initial_kernel_size: int,
    bucket_size: int,
) -> Tuple[List[gp.Var], List[gp.Var], List[List[gp.Var]]]:
    """
    Solve the LP relaxation and partition all discrete variables into Kernel and Buckets.

    Variables are ranked based on their fractional values (closer to 1.0 is better).

    Args:
        model (gp.Model): The model instance.
        x (Dict): Flow variables.
        y (Dict): Selection variables.
        initial_kernel_size (int): Size of the primary search space (the Kernel).
        bucket_size (int): Size of secondary increments (Buckets).

    Returns:
        Tuple: (kernel_vars, remaining_vars, buckets)
    """
    model.optimize()
    if model.Status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or model.SolCount == 0:
        return [], [], []

    # 1. Rank variables by fractional value
    all_vars = []
    for var in x.values():
        all_vars.append((var, var.X))
    for var in y.values():
        all_vars.append((var, var.X))

    all_vars.sort(key=lambda item: item[1], reverse=True)

    # 2. Switch types to BINARY for the remainder of the search
    for v_obj, _ in all_vars:
        v_obj.vtype = GRB.BINARY

    # 3. Partition into sets
    kernel_vars = [v for v, _ in all_vars[:initial_kernel_size]]
    remaining_vars = [v for v, _ in all_vars[initial_kernel_size:]]
    buckets = [remaining_vars[i : i + bucket_size] for i in range(0, len(remaining_vars), bucket_size)]

    return kernel_vars, remaining_vars, buckets


def _solve_ks_iterations(
    model: gp.Model,
    kernel_vars: List[gp.Var],
    buckets: List[List[gp.Var]],
    remaining_vars: List[gp.Var],
    time_limit: float,
    mip_limit_nodes: int,
) -> Set[gp.Var]:
    """
    Perform the iterative improvement phase by solving sub-problems.

    Each iteration adds one bucket of variables to the search space. If the objective
    improves, all variables used in the new solution are promoted to the Kernel permanently.
    Otherwise, unused variables in the bucket are re-fixed to 0.

    Args:
        model: Gurobi model.
        kernel_vars: Seed variables.
        buckets: List of variable subsets.
        remaining_vars: All variables outside the initial kernel.
        time_limit: Total runtime budget.
        mip_limit_nodes: Branch-and-bound node limit per iteration.

    Returns:
        Set: All variables that took positive values in the best found solution.
    """
    # Fix everything to zero initially
    for v in remaining_vars:
        v.ub = 0

    best_obj = -float("inf")
    used_vars = set()
    start_time = model.Runtime

    # 1. Solve INITIAL KERNEL
    model.setParam("TimeLimit", max(1.0, (time_limit / (len(buckets) + 1))))
    model.setParam("NodeLimit", mip_limit_nodes)
    model.optimize()

    if model.SolCount > 0:
        best_obj = model.ObjVal
        used_vars = {v for v in kernel_vars if v.X > 0.5}

    # 2. Iterative BUCKET refinement
    for bucket in buckets:
        # Check time budget
        if (model.Runtime - start_time) > time_limit:
            break

        # Dynamically unfix the current bucket
        for v in bucket:
            v.ub = 1

        model.optimize()

        if model.SolCount > 0:
            if model.ObjVal > best_obj:
                best_obj = model.ObjVal
                # If improvement, capture NEW useful variables
                for v in bucket:
                    if v.X > 0.5:
                        used_vars.add(v)

            # Fix unused variables in the bucket back to zero to keep sub-MIP small
            for v in bucket:
                if v.X < 0.5:
                    v.ub = 0
        else:
            # Revert unfixing if no better solution found
            for v in bucket:
                v.ub = 0

    return used_vars


def _reconstruct_tour(
    num_nodes: int, x: Dict[Tuple[int, int], gp.Var], dist_matrix: np.ndarray
) -> Tuple[List[int], float]:
    """
    Convert the resulting binary flow variables into a sequence of nodes (a tour).

    Args:
        num_nodes: Total number of nodes in graph.
        x: Flow variables from the Gurobi model.
        dist_matrix: Matrix used for distance calculation.

    Returns:
        Tuple: (tour list, total routing cost)
    """
    # 1. Extract activated edges
    active_edges = [(i, j) for (i, j), var in x.items() if var.X > 0.5]
    if not active_edges:
        return [0, 0], 0.0

    # 2. Build adjacency map for path reconstruction
    adj = {i: [] for i in range(num_nodes)}
    for i, j in active_edges:
        adj[i].append(j)

    # 3. Traverse the path starting at depot 0
    full_tour = [0]
    current = 0
    visited_nodes = {0}
    while True:
        if not adj[current]:
            break
        nx_node = adj[current].pop(0)
        # Handle potential cycle/convergence to zero
        if nx_node in visited_nodes and nx_node != 0:
            break
        full_tour.append(nx_node)
        visited_nodes.add(nx_node)
        current = nx_node
        if current == 0:
            break

    # 4. Ensure it closes at the depot
    if full_tour[-1] != 0:
        full_tour.append(0)

    # 5. Calculate final cost
    routing_cost = sum(dist_matrix[i][j] for i, j in active_edges)
    return full_tour, float(routing_cost)


def run_kernel_search_gurobi(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    mandatory_nodes: List[int],
    initial_kernel_size: int = 50,
    bucket_size: int = 20,
    max_buckets: int = 10,
    time_limit: float = 300.0,
    mip_limit_nodes: int = 5000,
    mip_gap: float = 0.01,
    seed: int = 42,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[int], float, float]:
    """
    Entry point to solve the VRPP via Kernel Search matheuristic.

    Orchestrates the model setup, variable partitioning, iterative improvement,
    and final tour reconstruction.

    Args:
        dist_matrix: Pairwise distance matrix.
        wastes: Node demands/waste levels.
        capacity: Vehicle capacity.
        R: Revenue multiplier for collected waste.
        C: Cost multiplier for distance.
        mandatory_nodes: Nodes that must be visited.
        initial_kernel_size: Number of variables in the primary search space.
        bucket_size: Number of variables explored per iteration.
        max_buckets: Maximum iterations (bucket solves).
        time_limit: Total wall-clock runtime limit (seconds).
        mip_limit_nodes: Maximum branch-and-bound nodes per sub-MIP solve.
        mip_gap: Targeted optimality gap for each sub-problem.
        seed: Random seed for solver reproducibility.
        env: Optional pre-configured Gurobi environment.
        recorder: Optional utility for tracking search progress/statistics.

    Returns:
        Tour, MILP Objective value, and physical Routing Cost.
    """
    # 1. Initialize Model
    model = gp.Model("KS_VRPP", env=env) if env else gp.Model("KS_VRPP")
    model.setParam("OutputFlag", 0)  # Deactivate verbose output
    model.setParam("Seed", seed)
    model.setParam("MIPGap", mip_gap)

    # Phase 1: Structural Setup
    x, y, _ = _setup_ks_model(model, dist_matrix, wastes, capacity, R, C, mandatory_nodes)

    # Phase 2: LP Relaxation & Variable Partitioning
    # This phase creates the ranking of variables for the Kernel and Buckets.
    kernel_vars, remaining_vars, buckets = _get_partitioned_vars(model, x, y, initial_kernel_size, bucket_size)
    if not kernel_vars:
        # Fallback if LP solve failed to find even partial feasibility
        return [0, 0], 0.0, 0.0

    # Limit search space by max_buckets parameter
    buckets = buckets[:max_buckets]

    # Phase 3: Iterative Search (Incremental MIPs)
    # The kernel grows and unused variables are pruned in each step.
    used_vars = _solve_ks_iterations(model, kernel_vars, buckets, remaining_vars, time_limit, mip_limit_nodes)

    # Phase 4: Final extraction and tour reconstruction
    if not used_vars:
        return [0, 0], 0.0, 0.0

    # Unfix all variables identified as useful for a final "polishing" solve
    for v in used_vars:
        v.ub = 1
    model.optimize()

    if model.SolCount == 0:
        return [0, 0], 0.0, 0.0

    tour, cost = _reconstruct_tour(len(dist_matrix), x, dist_matrix)

    # 5. Analytics tracking
    if recorder:
        recorder.record(engine="kernel_search", obj_val=model.ObjVal, cost=cost, solved=1)

    return tour, float(model.ObjVal), float(cost)
