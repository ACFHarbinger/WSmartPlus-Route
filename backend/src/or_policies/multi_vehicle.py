from fast_tsp import compute_cost
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def find_routes(dist_mat, demands, max_caps, to_collect, n_vehicles, depot):
    to_collect_tmp = [0]
    to_collect_tmp.extend(list(to_collect))
    distancesC = dist_mat[to_collect_tmp, :][:, to_collect_tmp]

    manager = pywrapcp.RoutingIndexManager(len(distancesC), n_vehicles, depot)
    routing = pywrapcp.RoutingModel(manager)

    # Create and register a transit callback.
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex.
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return distancesC[from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)

    # Define cost of each arc.
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # Add Capacity constraint.
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex.
        from_node = manager.IndexToNode(from_index)
        return demands[from_node]

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        max_caps,  # vehicle maximum capacities
        True,  # start cumul to zero
        "Capacity",
    )

    # Setting first solution heuristic.
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(2)

    solution = routing.SolveWithParameters(search_parameters)
    tours, costs, _, _ = get_solution_costs(demands, n_vehicles, manager, routing, solution, distancesC)
    return tours, costs


def get_solution_costs(demands, n_vehicles, manager, routing, solution, distancesC):
    dist_ls = []
    load_ls = []
    tours = []
    costs = []
    for vehicle_id in range(n_vehicles):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        route = [index]
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += demands[node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
            route.append(index)
        dist_ls.append(route_distance)
        load_ls.append(route_load)
        tours.append(route)
        costs.append(compute_cost(route, distancesC))
    return tours, costs, dist_ls, load_ls