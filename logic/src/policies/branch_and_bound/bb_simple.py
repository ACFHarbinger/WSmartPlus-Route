"""
Simplified Branch-and-Bound solver that delegates to Gurobi's built-in B&B.

This replaces the custom BB implementation which was solving full MIPs at each node,
causing exponential slowdown. Instead, we just set up the model once and let Gurobi
handle the branch-and-bound tree internally.
"""

from typing import Any, Dict, List, Optional, Set, Tuple

import gurobipy as gp
import networkx as nx
import numpy as np
from gurobipy import GRB, quicksum

from logic.src.tracking.viz_mixin import PolicyStateRecorder


def _dfj_callback(model, where):
    """DFJ subtour elimination callback for Gurobi's built-in B&B."""
    if where == GRB.Callback.MIPSOL:
        x_vars = model._x_vars
        num_nodes = model._num_nodes

        # Build graph from active edges
        G = nx.Graph()
        G.add_nodes_from(range(num_nodes))

        for (i, j), var in x_vars.items():
            val = model.cbGetSolution(var)
            if val > 0.5:
                G.add_edge(i, j)

        # Find connected components
        components = list(nx.connected_components(G))

        # Add cuts for subtours
        for component in components:
            if 0 not in component and len(component) >= 2:
                subtour_edges = []
                for i in component:
                    for j in component:
                        if i != j and (i, j) in x_vars:
                            subtour_edges.append(x_vars[(i, j)])

                if subtour_edges:
                    model.cbLazy(quicksum(subtour_edges) <= len(component) - 1)


def _setup_bb_model(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    must_go_indices: Set[int],
    time_limit: float,
    mip_gap: float,
    seed: int,
    env: Optional[gp.Env] = None,
) -> Tuple[gp.Model, Dict[Tuple[int, int], gp.Var], Dict[int, gp.Var]]:
    """Set up the Gurobi model for VRPP."""
    num_nodes = len(dist_matrix)
    nodes = range(num_nodes)
    customers = list(range(1, num_nodes))

    model = gp.Model("BB_Simple", env=env) if env else gp.Model("BB_Simple")
    model.setParam("OutputFlag", 0)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", mip_gap)
    model.setParam("Seed", seed)
    model.Params.LazyConstraints = 1

    # Decision variables (BINARY)
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"x_{i}_{j}")

    y = {}
    for i in customers:
        y[i] = model.addVar(lb=0, ub=1, vtype=GRB.BINARY, name=f"y_{i}")

    # Objective and constraints set in run_bb_simple to keep params local if needed,
    # but we can pass R and C here or just return the variables.
    return model, x, y


def _extract_routes_from_adj(adj: Dict[int, List[int]], num_nodes: int) -> List[List[int]]:
    """Extract routes starting from the depot using an adjacency list."""
    routes = []
    for start in adj[0]:
        route = [start]
        current = start
        visited = {0, start}

        while current != 0 and adj[current]:
            next_node = adj[current][0]
            if next_node in visited and next_node != 0:
                break
            if next_node != 0:
                route.append(next_node)
                visited.add(next_node)
            current = next_node
            if current == 0:
                break

        if route:
            routes.append(route)
    return routes


def run_bb_simple(
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    must_go_indices: Optional[Set[int]] = None,
    env: Optional[gp.Env] = None,
    recorder: Optional[PolicyStateRecorder] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve VRPP using Gurobi's built-in branch-and-bound with DFJ lazy constraints.

    This is a simplified version that doesn't implement custom BB logic, but instead
    delegates to Gurobi's highly optimized internal B&B engine.

    Args:
        dist_matrix: Distance matrix (n x n)
        wastes: Waste/profit at each node
        capacity: Vehicle capacity
        R: Revenue multiplier
        C: Cost multiplier
        values: Configuration dict with time_limit, mip_gap, etc.
        must_go_indices: Nodes that must be visited
        env: Optional Gurobi environment
        recorder: Optional state recorder

    Returns:
        Tuple of (routes, objective_value)
    """
    num_nodes = len(dist_matrix)
    nodes = range(num_nodes)
    customers = list(range(1, num_nodes))
    must_go_indices = must_go_indices or set()

    # Extract configuration
    time_limit = values.get("time_limit", 60.0)
    mip_gap = values.get("mip_gap", 0.01)
    seed = values.get("seed", 42)

    # Setup basic model structure
    model, x, y = _setup_bb_model(dist_matrix, wastes, must_go_indices, time_limit, mip_gap, seed, env)

    # Objective: Maximize (Revenue - Cost)
    travel_cost = quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)
    revenue = quicksum(wastes.get(i, 0) * R * y[i] for i in customers)
    model.setObjective(revenue - travel_cost, GRB.MAXIMIZE)

    # Flow balance constraints
    for i in customers:
        model.addConstr(quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"out_{i}")
        model.addConstr(quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"in_{i}")

    # Depot constraints
    model.addConstr(quicksum(x[0, j] for j in customers) <= 1, name="depot_out")
    model.addConstr(quicksum(x[j, 0] for j in customers) == quicksum(x[0, j] for j in customers), name="depot_balance")

    # Mandatory nodes
    for i in must_go_indices:
        if i in y:
            model.addConstr(y[i] == 1, name=f"must_go_{i}")

    # Store for callback
    model._x_vars = x
    model._num_nodes = num_nodes

    # Solve with DFJ callback
    model.optimize(_dfj_callback)

    # Extract solution
    if model.SolCount == 0:
        return [], 0.0

    active_edges = [(i, j) for (i, j), var in x.items() if var.X > 0.5]
    if not active_edges:
        return [], 0.0

    # Build adjacency list and extract routes
    adj = {i: [] for i in nodes}
    for i, j in active_edges:
        adj[i].append(j)

    routes = _extract_routes_from_adj(adj, num_nodes)

    # Record stats
    if recorder:
        recorder.record(
            engine="bb_simple",
            obj_val=model.ObjVal,
            nodes_explored=int(model.NodeCount),
            time=model.Runtime,
            solved=1,
        )

    return routes, float(model.ObjVal)
