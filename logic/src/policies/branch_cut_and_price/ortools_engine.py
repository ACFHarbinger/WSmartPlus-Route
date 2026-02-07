"""
OR-Tools engine for Branch-Cut-and-Price module.
"""

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def run_bcp_ortools(dist_matrix, demands, capacity, R, C, values, must_go_indices=None):
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
