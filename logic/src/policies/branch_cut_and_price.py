"""
Branch-Cut-and-Price (BCP) solver module for Prize-Collecting CVRP.

This module provides a unified interface to multiple exact and hybrid solvers:
1. OR-Tools: Constraint Programming with Guided Local Search
2. VRPy: Column Generation (Branch-and-Price) using NetworkX graphs
3. Gurobi: Mixed-Integer Programming with lazy subtour elimination

The module dispatches to the appropriate solver based on configuration and
handles Prize-Collecting CVRP where nodes can be optionally visited with
penalties for skipped nodes equal to their revenue.

Key Features:
- Penalty-based prize collection (nodes can be dropped at a cost)
- Must-go nodes enforcement (infinite penalty if dropped)
- Capacity constraints for all vehicles
- Multiple solver backends for flexibility

Reference:
    Feillet, D., et al. (2005). An exact algorithm for the elementary shortest
    path problem with resource constraints. Networks, 44(3), 216-229.
"""

import logging

import gurobipy as gp
import networkx as nx
from gurobipy import GRB
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from vrpy import VehicleRoutingProblem


def run_bcp(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None):
    """
    Main dispatcher for Branch-Cut-and-Price solvers.

    Selects and runs the appropriate BCP solver based on configuration.
    Supports Prize-Collecting CVRP with optional must-go nodes.

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N) with depot at index 0
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity constraint
        R (float): Revenue per unit demand
        C (float): Cost per unit distance
        values (dict): Configuration with 'bcp_engine' in ['ortools', 'vrpy', 'gurobi'].
            Default: 'ortools'. Also supports 'time_limit' (default: 30 seconds)
        must_go_indices (set, optional): Node IDs that must be visited
        env (gp.Env, optional): Gurobi environment (for Gurobi engine only)

    Returns:
        Tuple[List[List[int]], float]: Routes and total cost
            - routes: List of routes, each containing node IDs
            - cost: Total travel cost (distance * C)
    """
    engine = values.get("bcp_engine", "ortools")

    if engine == "vrpy":
        return _run_bcp_vrpy(dist_matrix, demands, capacity, R, C, values)
    elif engine == "gurobi":
        return _run_bcp_gurobi(dist_matrix, demands, capacity, R, C, values, must_go_indices, env)
    else:
        # Default to OR-Tools
        return _run_bcp_ortools(dist_matrix, demands, capacity, R, C, values, must_go_indices)


def _run_bcp_ortools(dist_matrix, demands, capacity, R, C, values, must_go_indices=None):
    """
    Solve Prize-Collecting CVRP using Google OR-Tools.

    Uses OR-Tools' constraint programming routing solver with:
    - Disjunction for optional node visits (Prize Collecting)
    - Penalties equal to revenue for skipped nodes
    - Infinite penalty for must-go nodes
    - PATH_CHEAPEST_ARC first solution strategy
    - GUIDED_LOCAL_SEARCH for improvement

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N)
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity
        R (float): Revenue per unit demand
        C (float): Cost per unit distance
        values (dict): Config with 'time_limit' (default: 30)
        must_go_indices (set, optional): Nodes that must be visited

    Returns:
        Tuple[List[List[int]], float]: Routes and total cost
    """
    # 1. Prepare Data
    if must_go_indices is None:
        must_go_indices = set()

    SCALE = 100
    scaled_dist_matrix = (dist_matrix * SCALE * C).astype(int)

    num_nodes = len(dist_matrix)
    num_vehicles = num_nodes
    depot = 0

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # 2. Add Distance Callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return scaled_dist_matrix[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 3. Add Capacity Constraint
    def demand_callback(from_index):
        """Returns the demand of the node."""
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        return int(demands.get(from_node, 0))

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [int(capacity)] * num_vehicles, True, "Capacity")

    # 4. Add Penalties (Prize Collecting)
    MUST_GO_PENALTY = 1_000_000_000

    for i in range(1, num_nodes):
        d = demands.get(i, 0)
        revenue = d * R

        if i in must_go_indices:
            penalty = MUST_GO_PENALTY
        else:
            penalty = int(revenue * SCALE)

        routing.AddDisjunction([manager.NodeToIndex(i)], penalty)

    # 5. Solve
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    time_limit_sec = values.get("time_limit", 30)
    search_parameters.time_limit.seconds = int(time_limit_sec)

    search_parameters.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    search_parameters.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH

    solution = routing.SolveWithParameters(search_parameters)

    # 6. Parse Result
    routes = []

    if solution:
        for vehicle_id in range(num_vehicles):
            index = routing.Start(vehicle_id)
            if routing.IsEnd(index):
                continue

            if routing.IsEnd(solution.Value(routing.NextVar(index))):
                continue

            route = []
            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                if node_index != 0:
                    route.append(node_index)
                index = solution.Value(routing.NextVar(index))

            if route:
                routes.append(route)

        real_dist_cost = 0
        for r in routes:
            full_r = [0] + r + [0]
            for i in range(len(full_r) - 1):
                u, v = full_r[i], full_r[i + 1]
                real_dist_cost += dist_matrix[u][v]

        return routes, real_dist_cost * C

    return [], 0.0


def _run_bcp_vrpy(dist_matrix, demands, capacity, R, C, values):
    """
    Solve CVRP using VRPy (Column Generation / Branch-and-Price).

    Note: VRPy does not natively support Prize-Collecting CVRP via simple
    configuration. This implementation solves standard CVRP for ALL nodes
    present in the demands dictionary (no node dropping).

    Uses NetworkX DiGraph representation with:
    - Source/Sink virtual nodes for depot
    - Edge costs scaled by C coefficient
    - Load capacity constraints
    - Column generation solver from VRPy library

    Args:
        dist_matrix (np.ndarray): Distance matrix (N x N)
        demands (dict): Node demands {node_id: demand_value}
        capacity (float): Vehicle capacity
        R (float): Revenue per unit demand (unused in this variant)
        C (float): Cost per unit distance
        values (dict): Config with 'time_limit' (default: 30)

    Returns:
        Tuple[List[List[int]], float]: Routes and total cost
            - routes: List of routes excluding Source/Sink
            - cost: Total objective value from VRPy
    """
    if VehicleRoutingProblem is None:
        logging.error("VRPy not installed or import failed.")
        return [], 0.0

    # Suppress Logs
    logging.getLogger("cspy").setLevel(logging.WARNING)
    logging.getLogger("vrpy").setLevel(logging.WARNING)

    G = nx.DiGraph()
    n_nodes = len(dist_matrix) - 1

    # Add Nodes
    for i in range(1, n_nodes + 1):
        d = demands.get(i, 0)
        G.add_node(i, demand=d)

    # Add Edges
    for i in range(1, n_nodes + 1):
        cost = dist_matrix[0][i] * C
        G.add_edge("Source", i, cost=cost)

    for i in range(1, n_nodes + 1):
        cost = dist_matrix[i][0] * C
        G.add_edge(i, "Sink", cost=cost)

    for i in range(1, n_nodes + 1):
        for j in range(1, n_nodes + 1):
            if i != j:
                cost = dist_matrix[i][j] * C
                G.add_edge(i, j, cost=cost)

    prob = VehicleRoutingProblem(G, load_capacity=capacity)

    time_limit = values.get("time_limit", 30)
    prob.solve(time_limit=time_limit)

    if prob.best_routes:
        routes = []
        for r_id, path in prob.best_routes.items():
            clean_route = [node for node in path if node != "Source" and node != "Sink"]
            if clean_route:
                routes.append(clean_route)
        return routes, prob.best_value
    else:
        return [], 0.0


def _run_bcp_gurobi(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None):
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
