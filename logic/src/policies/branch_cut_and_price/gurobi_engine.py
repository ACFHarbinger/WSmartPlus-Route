"""
Gurobi engine for Branch-Cut-and-Price module.
"""

import gurobipy as gp
from gurobipy import GRB


def _setup_model(values, env):
    """Initialize Gurobi model with parameters."""
    model = gp.Model("CVRP", env=env) if env else gp.Model("CVRP")
    model.setParam("TimeLimit", values.get("time_limit", 30))
    model.setParam("MIPGap", 0.05)
    return model


def _create_variables(model, nodes, customers, demands, capacity):
    """Create decision variables x, y, u."""
    # x[i,j]: 1 if edge (i,j) used.
    x = {}
    for i in nodes:
        for j in nodes:
            if i != j:
                x[i, j] = model.addVar(vtype=GRB.BINARY, name=f"x_{i}_{j}")

    # y[i]: 1 if node i is visited (Prize Collecting)
    y = {}
    for i in customers:
        y[i] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}")

    # u[i]: load after visiting node i (MTZ)
    u = {}
    for i in customers:
        u[i] = model.addVar(lb=demands.get(i, 0), ub=capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    return x, y, u


def _create_objective(model, x, y, dist_matrix, demands, nodes, customers, R, C, must_go_indices):
    """Set the objective function: Minimize Cost + Penalties."""
    travel_cost = gp.quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)

    revenue_penalty = 0
    for i in customers:
        d = demands.get(i, 0)
        rev = d * R
        if i in must_go_indices:
            # Must Visit constraint
            model.addConstr(y[i] == 1, name=f"must_visit_{i}")
        else:
            # Penalty if y[i] is 0 -> (1 - y[i]) * rev
            revenue_penalty += (1 - y[i]) * rev

    model.setObjective(travel_cost + revenue_penalty, GRB.MINIMIZE)


def _add_constraints(model, x, y, u, nodes, customers, demands, capacity):
    """Add flow conservation and capacity constraints."""
    # Flow Conservation
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"flow_out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"flow_in_{i}")

    # MTZ Constraints
    for i in customers:
        for j in customers:
            if i != j:
                d_j = demands.get(j, 0)
                model.addConstr(u[j] >= u[i] + d_j - capacity * (1 - x[i, j]), name=f"mtz_{i}_{j}")


def _extract_solution(model, x, nodes):
    """Extract optimal routes from the model."""
    if model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT] or model.SolCount == 0:
        return [], 0.0

    routes = []
    # Build adjacency list
    adj = {i: [] for i in nodes}  # type: ignore[var-annotated]
    for i in nodes:
        for j in nodes:
            if i != j and x[i, j].X > 0.5:
                adj[i].append(j)

    # Trace routes
    for start_node in adj[0]:
        route = []
        curr = start_node
        while curr != 0:
            route.append(curr)
            if not adj[curr]:
                break
            curr = adj[curr][0]
        routes.append(route)

    return routes, model.objVal


def run_bcp_gurobi(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None):
    """
    Solve Prize-Collecting CVRP using Gurobi MIP solver.

    Implements 2-index flow formulation with MTZ constraints.
    """
    if must_go_indices is None:
        must_go_indices = set()

    # Identifying Customer Nodes: Indices 1..N
    N = len(dist_matrix) - 1
    customers = [i for i in range(1, N + 1)]
    nodes = [0] + customers

    # 1. Setup Model
    model = _setup_model(values, env)

    # 2. Create Variables
    x, y, u = _create_variables(model, nodes, customers, demands, capacity)

    # 3. Objective
    _create_objective(model, x, y, dist_matrix, demands, nodes, customers, R, C, must_go_indices)

    # 4. Constraints
    _add_constraints(model, x, y, u, nodes, customers, demands, capacity)

    # 5. Optimize
    model.optimize()

    # 6. Extract Solution
    return _extract_solution(model, x, nodes)
