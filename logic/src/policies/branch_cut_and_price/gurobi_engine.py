"""
Gurobi engine for Branch-Cut-and-Price module.
"""

import gurobipy as gp
from gurobipy import GRB


def run_bcp_gurobi(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None):
    """
    Solve Prize-Collecting CVRP using Gurobi MIP solver.

    Implements 2-index flow formulation with:
    - Binary edge variables x[i,j] for routing
    - Binary visit variables y[i] for Prize Collecting
    - MTZ (Miller-Tucker-Zemlin) constraints for capacity and subtour elimination
    - Objective: Minimize travel cost + penalties for dropped nodes
    - Must-go nodes enforced via y[i] = 1 constraints

    Formulation:
        min: sum(dist[i][j] * C * x[i,j]) + sum((1 - y[i]) * revenue[i])
        s.t.: Flow conservation, capacity (MTZ), visit logic

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N)
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity
        R (float): Revenue per unit demand
        C (float): Cost per unit distance
        values (dict): Config with 'time_limit' (default: 30), 'MIPGap' (default: 0.05)
        must_go_indices (set, optional): Nodes that must be visited
        env (gp.Env, optional): Gurobi environment for license control

    Returns:
        Tuple[List[List[int]], float]: Routes and objective value
    """
    if must_go_indices is None:
        must_go_indices = set()

    # Identifying Customer Nodes
    # Indices 1..N
    N = len(dist_matrix) - 1
    customers = [i for i in range(1, N + 1)]
    nodes = [0] + customers

    # Filter: Only include Customers that are in demand set?
    # Usually passed demands covers the candidates.
    # Note on "Prize Collecting": Standard MIP visits ALL customers unless we add logic.
    # User requested BCP *Variations*. OR-Tools is PC-CVRP.
    # VRPy is CVRP.
    # Gurobi should ideally be PC-CVRP to match functionality, or CVRP.
    # Adding PC-CVRP logic to MIP is easy (make visit variable).

    model = gp.Model("CVRP", env=env) if env else gp.Model("CVRP")
    model.setParam("TimeLimit", values.get("time_limit", 30))
    model.setParam("MIPGap", 0.05)

    # Variables
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

    # Demand satisfaction logic for dropped nodes?
    # PC-CVRP Objective: Maximize Profit = Sum(Revenue * y[i]) - Sum(Cost * Dist * x[i,j])
    # Or Minimize Cost_Dist + Penalty_Dropped

    # Let's match OR-Tools Logic: Minimize Cost = Dist*x + Penalties(Dropped)
    # Penalty_Dropped = Revenue

    # Objective
    travel_cost = gp.quicksum(dist_matrix[i][j] * C * x[i, j] for i in nodes for j in nodes if i != j)

    revenue_penalty = 0
    # Add dropped penalties
    for i in customers:
        d = demands.get(i, 0)
        rev = d * R
        # if must_go: infinity penalty if dropped (y=0)
        if i in must_go_indices:
            # Must Visit constraint
            model.addConstr(y[i] == 1, name=f"must_visit_{i}")
        else:
            # Penalty if y[i] is 0 -> (1 - y[i]) * rev
            revenue_penalty += (1 - y[i]) * rev

    model.setObjective(travel_cost + revenue_penalty, GRB.MINIMIZE)

    # Constraints

    # Flow Conservation
    # sum(x[i,j]) = y[i] (out)
    # sum(x[j,i]) = y[i] (in)
    for i in customers:
        model.addConstr(gp.quicksum(x[i, j] for j in nodes if i != j) == y[i], name=f"flow_out_{i}")
        model.addConstr(gp.quicksum(x[j, i] for j in nodes if i != j) == y[i], name=f"flow_in_{i}")

    # Depot Flow
    # K vehicles used
    # sum(x[0,j]) <= N
    # We leave number of vehicles free (minimized by cost implicitly if fixed costs exist, or just valid routing)

    # Capacity Constraints (MTZ or Flow)
    # Subtours & Capacity
    # Lazy Constraints are best for subtours.
    # But for Capacity, simple MTZ is easier to implement quickly for a variant.
    # u[i] = load after visiting node i
    u = {}
    for i in customers:
        u[i] = model.addVar(lb=demands.get(i, 0), ub=capacity, vtype=GRB.CONTINUOUS, name=f"u_{i}")

    for i in customers:
        for j in customers:
            if i != j:
                # MTZ: u[j] >= u[i] + d[j] - Q(1-x[ij])
                # Only strictly binding if x[ij]=1
                d_j = demands.get(j, 0)
                model.addConstr(u[j] >= u[i] + d_j - capacity * (1 - x[i, j]), name=f"mtz_{i}_{j}")

    model.optimize()

    # Parse solution
    if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT] and model.SolCount > 0:
        routes = []
        # Reconstruct routes from x
        # Find edges starting from 0

        # Build adjacency list
        adj = {i: [] for i in nodes}
        for i in nodes:
            for j in nodes:
                if i != j and x[i, j].X > 0.5:
                    adj[i].append(j)

        # Trace routes
        # Each departure from 0 is a route
        for start_node in adj[0]:
            route = []
            curr = start_node
            while curr != 0:
                route.append(curr)
                if not adj[curr]:
                    break  # Should not happen in valid flow
                curr = adj[curr][0]  # Should be 1 outgoing
            routes.append(route)

        return routes, model.objVal

    return [], 0.0
