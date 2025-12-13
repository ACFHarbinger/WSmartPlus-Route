
import pyvrp
import numpy as np

from fast_tsp import compute_cost
from pyvrp.stop import MaxRuntime
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2


def find_routes(dist_mat, demands, max_caps, to_collect, n_vehicles, coords=None, depot=0):
    """
    Find routes using PyVRP.
    
    Args:
        dist_mat (np.ndarray): Distance matrix.
        demands (np.ndarray): Demand of each node (index matches dist_mat).
        max_caps (int): Capacity per vehicle.
        to_collect (list/array): Indices of bins to collect (1-based, usually).
        n_vehicles (int): Number of vehicles.
        coords (pd.DataFrame, optional): Coordinates of nodes.
        depot (int): Depot index (usually 0).
        
    Returns:
        tuple: (tour, cost)
            tour: Flattened list of nodes visited, e.g. [0, 1, 2, 0, 3, 0].
            cost: Total cost.
    """
    # Mapping: subset index -> original index
    # Internal PyVRP: 0 is depot. 1..N are clients.
    # We map indices 0..N of our subset problems to original indices.
    subset_indices = [depot] + list(to_collect)
    
    # Create sub-matrices
    sub_dist = dist_mat[np.ix_(subset_indices, subset_indices)]
    # sub_demands is not direct slicing because demands is 0-indexed (bins) and subset_indices are 1-indexed (nodes)
    
    # Build ProblemData
    # Clients (indices 1..N in subset)
    clients_list = []
    for i in range(1, len(subset_indices)):
         original_idx = subset_indices[i]
         # Demand for node original_idx is at demands[original_idx - 1]
         # assuming depot is 0 and demands are for nodes 1..N
         d = int(demands[original_idx - 1])
         # Use delivery for demand
         clients_list.append(pyvrp.Client(x=0, y=0, delivery=d))
         
    depots_list = [pyvrp.Depot(x=0, y=0)]
    
    # If n_vehicles is 0, we treat it as unlimited -> set to number of clients
    if n_vehicles == 0:
        n_vehicles = len(clients_list)

    vehicle_types_list = [pyvrp.VehicleType(capacity=int(max_caps), num_available=n_vehicles)]
    
    # Matrix
    # PyVRP expects List[numpy.ndarray[int]]
    matrix = sub_dist.astype(int)
    
    data = pyvrp.ProblemData(
        clients=clients_list,
        depots=depots_list,
        vehicle_types=vehicle_types_list,
        distance_matrices=[matrix],
        duration_matrices=[matrix]
    )
    
    # Solve
    # Seed is optional but good for determinism.
    res = pyvrp.solve(data, stop=MaxRuntime(2.0), seed=42)
    
    tour_flat = []
    total_cost = res.cost()
    
    # Parse result
    # res.best.routes() -> list of Route
    for route in res.best.routes():
        # Each route starts and ends at depot implicitly or explicitly? 
        # PyVRP Route object iterates over clients.
        
        # Start new route with depot
        if not tour_flat:
            tour_flat.append(0)
            
        # Add clients
        for node_idx in route: 
            # node_idx is 1-based index in `clients` list of ProblemData?
            # PyVRP convention: 0 is depot, 1 is first client.
            original_node = subset_indices[node_idx]
            tour_flat.append(original_node)
            
        # End route with depot
        tour_flat.append(0)
        
    # Example format: [0, 1, 2, 0, 3, 4, 0]
    # My logic:
    # R1: [1, 2]. -> adds 0, 1, 2, 0. list: [0, 1, 2, 0]
    # R2: [3, 4]. -> adds 3, 4, 0. list: [0, 1, 2, 0, 3, 4, 0] (Wait, check loop)
    
    # Logic in loop:
    # if not tour_flat: append(0) -> [0]
    # append nodes -> [0, 1, 2]
    # append 0 -> [0, 1, 2, 0]
    # Next route:
    # if not tour_flat (false)
    # append nodes -> [0, 1, 2, 0, 3, 4]
    # append 0 -> [0, 1, 2, 0, 3, 4, 0]
    # Correct.
    
    return tour_flat, total_cost


def find_routes_ortools(dist_mat, demands, max_caps, to_collect, n_vehicles, depot):
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