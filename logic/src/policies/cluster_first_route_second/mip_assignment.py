"""
Mixed-Integer Programming Assignment Module for Fisher & Jaikumar Clustering.

This module implements an exact optimization approach for assigning nodes to seed
clusters using Gurobi's MIP solver. The formulation maximizes total profit
(revenue - cost) subject to capacity and assignment constraints.

This is an extension of the original Fisher & Jaikumar algorithm that uses
exact optimization instead of greedy heuristics for the assignment phase.

Reference:
    Fisher, M. L., & Jaikumar, R. (1981). "A generalized assignment heuristic
    for vehicle routing". Networks, 11(2), 109-124.
"""

from typing import Dict, List

import numpy as np

try:
    import gurobipy as gp
    from gurobipy import GRB

    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False


def assign_exact_mip(
    seeds: List[int],
    must_go: List[int],
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    distance_matrix: np.ndarray,
    time_limit: float = 60.0,
    objective: str = "minimize_cost",
) -> List[List[int]]:
    """
    Assign nodes to seeds by solving the Generalized Assignment Problem (GAP)
    as an exact Mixed-Integer Program using Gurobi.

    Supports two objective modes:
    1. **minimize_cost** (default): Minimize total insertion cost for benchmark compliance
    2. **maximize_profit**: Maximize profit (revenue - cost) for simulation mode

    Mathematical Formulation (minimize_cost):
        Minimize: Σ_i Σ_k (c_ik * x_ik)
        Subject to:
            Σ_k x_ik <= 1                    ∀i ∈ must_go  (assignment constraint)
            Σ_i w_i * x_ik <= capacity       ∀k ∈ seeds    (capacity constraint)
            x_kk = 1                         ∀k ∈ seeds    (seed pre-assignment)
            x_ik ∈ {0, 1}                    ∀i, k         (binary variables)

    Mathematical Formulation (maximize_profit):
        Maximize: Σ_i Σ_k [(R * w_i) - (C * c_ik)] * x_ik
        Subject to: (same constraints as above)

        Where:
            x_ik = 1 if node i is assigned to seed k, 0 otherwise
            w_i = waste quantity at node i
            c_ik = insertion cost = d(0, i) + d(k, i) - d(k, 0)
            R = revenue per unit of waste
            C = cost per unit of distance

    Args:
        seeds: List of seed node indices (one per initial cluster).
        must_go: List of all nodes that need to be assigned.
        wastes: Dictionary mapping node indices to waste quantities.
        capacity: Maximum vehicle capacity.
        R: Revenue per unit of waste (used only in maximize_profit mode).
        C: Cost per unit of distance (used only in maximize_profit mode).
        distance_matrix: Pre-computed all-pairs distance matrix.
        time_limit: Maximum time in seconds for optimization (default: 60.0).
            The actual time used is 20% of this value.
        objective: Objective function ("minimize_cost" or "maximize_profit").
            Use "minimize_cost" for benchmark comparisons against A-VRP dataset.

    Returns:
        List of clusters, where each cluster is a list of node indices.
        Returns None if Gurobi is unavailable or MIP fails.

    Raises:
        ValueError: If objective is not "minimize_cost" or "maximize_profit".

    Note:
        - Requires gurobipy with valid license.
        - If Gurobi is unavailable or MIP fails, returns None to signal fallback needed.
        - Allocates 20% of total time_limit to the clustering phase.
        - For benchmark compliance with A-VRP dataset, use objective="minimize_cost".

    Example (Benchmark Mode):
        >>> seeds = [5, 12, 20]
        >>> must_go = list(range(1, 21))
        >>> wastes = {i: 10.0 for i in must_go}
        >>> capacity = 100.0
        >>> R, C = 5.0, 1.0
        >>> distance_matrix = np.random.rand(21, 21)
        >>> # Minimize cost for benchmark comparison
        >>> clusters = assign_exact_mip(seeds, must_go, wastes, capacity, R, C,
        ...                             distance_matrix, objective="minimize_cost")

    Example (Simulation Mode):
        >>> # Maximize profit for simulation
        >>> clusters = assign_exact_mip(seeds, must_go, wastes, capacity, R, C,
        ...                             distance_matrix, objective="maximize_profit")
    """
    # Validate objective parameter
    if objective not in ["minimize_cost", "maximize_profit"]:
        raise ValueError(
            f"Invalid objective: '{objective}'. "
            f"Must be 'minimize_cost' (benchmark mode) or 'maximize_profit' (simulation mode)."
        )

    # Check if Gurobi is available
    if not GUROBI_AVAILABLE:
        return None

    try:
        clusters: List[List[int]] = [[] for _ in range(len(seeds))]

        # 1. Precompute Fisher & Jaikumar insertion costs: c_ik = d(0, i) + d(i, k) - d(0, k)
        costs = {}
        for i in must_go:
            for k_idx, k in enumerate(seeds):
                costs[i, k_idx] = distance_matrix[0, i] + distance_matrix[i, k] - distance_matrix[0, k]

        # 2. Setup Gurobi Environment and Model
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)  # Silence solver output
        env.start()
        model = gp.Model("Fisher_Jaikumar_GAP", env=env)

        # 3. Decision Variables and Objective Function
        # x[i, k] = 1 if node i is assigned to seed/cluster k
        x = {}

        if objective == "minimize_cost":
            # Benchmark Mode: Minimize total insertion cost
            # Objective: Minimize Σ_i Σ_k (c_ik * x_ik)
            model.ModelSense = GRB.MINIMIZE

            for i in must_go:
                for k_idx in range(len(seeds)):
                    # Coefficient is insertion cost
                    cost_coeff = costs[i, k_idx]
                    x[i, k_idx] = model.addVar(vtype=GRB.BINARY, obj=cost_coeff, name=f"x_{i}_{k_idx}")

        else:  # objective == "maximize_profit"
            # Simulation Mode: Maximize profit (revenue - cost)
            # Objective: Maximize Σ_i Σ_k [(R * w_i) - (C * c_ik)] * x_ik
            model.ModelSense = GRB.MAXIMIZE

            for i in must_go:
                for k_idx in range(len(seeds)):
                    # Profit = Revenue from waste - Cost of insertion
                    profit = (wastes.get(i, 0.0) * R) - (costs[i, k_idx] * C)
                    x[i, k_idx] = model.addVar(vtype=GRB.BINARY, obj=profit, name=f"x_{i}_{k_idx}")

        # 4. Constraint: Assignment (Each node to AT MOST one cluster)
        for i in must_go:
            model.addConstr(gp.quicksum(x[i, k_idx] for k_idx in range(len(seeds))) <= 1, name=f"assign_{i}")

        # 5. Constraint: Vehicle Capacity
        for k_idx in range(len(seeds)):
            model.addConstr(
                gp.quicksum(wastes.get(i, 0.0) * x[i, k_idx] for i in must_go) <= capacity, name=f"cap_{k_idx}"
            )

        # 6. Constraint: Seeds must be assigned to their own clusters
        for k_idx, k in enumerate(seeds):
            if k in must_go:
                model.addConstr(x[k, k_idx] == 1, name=f"seed_{k}")

        # 7. Solve the GAP
        # Allocate 20% of the total time budget to the exact clustering phase
        gap_time_limit = max(1.0, time_limit * 0.20)
        model.setParam("TimeLimit", gap_time_limit)
        model.optimize()

        # 8. Extract Solution
        if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
            for i in must_go:
                for k_idx in range(len(seeds)):
                    if x[i, k_idx].X > 0.5:
                        clusters[k_idx].append(i)
                        break
        else:
            # MIP is infeasible or failed - signal fallback needed
            return None

        return clusters

    except Exception:
        # If Gurobi fails for any reason, signal fallback needed
        return None
