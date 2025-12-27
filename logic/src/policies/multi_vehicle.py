
import pyvrp
import numpy as np

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
        
    return tour_flat


def find_routes_ortools(dist_mat, demands, max_caps, to_collect, n_vehicles, coords=None, depot=0):
    # Mapping: subset index -> original index
    subset_indices = [depot] + list(to_collect)
    
    # Create sub-matrices and cast to int for OR-Tools/compute_cost
    distancesC = dist_mat[np.ix_(subset_indices, subset_indices)].astype(int)
    
    # Demands: map node idx to demand index (idx-1)
    sub_demands = [0] # Depot
    for idx in subset_indices[1:]:
        d = int(demands[idx - 1])
        sub_demands.append(d)
        
    # Unlimited logic
    if n_vehicles == 0:
        n_vehicles = len(to_collect)

    manager = pywrapcp.RoutingIndexManager(len(distancesC), n_vehicles, 0)
    routing = pywrapcp.RoutingModel(manager)

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(distancesC[from_node][to_node])

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        return int(sub_demands[from_node])

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        [int(max_caps)] * n_vehicles, 
        True,  # start cumul to zero
        "Capacity",
    )

    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.FromSeconds(2)

    solution = routing.SolveWithParameters(search_parameters)
    
    if not solution:
        return [0], 0

    # Ensure distancesC is passed as list of lists for compute_cost (unused now but keeps signal)
    tours_subset, costs_subset, _, _ = get_solution_costs(sub_demands, n_vehicles, manager, routing, solution, distancesC.tolist())
    
    # Flatten and Map Indices
    tour_flat = []
    for t in tours_subset:
        if not tour_flat:
            tour_flat.append(0)
             
        for node_idx in t:
            # node_idx is now NODE index (from get_solution_costs)
            if node_idx == 0: continue
            
            original_idx = subset_indices[node_idx]
            
            if original_idx == 0 and tour_flat[-1] == 0:
                continue
                
            tour_flat.append(original_idx)
             
        if tour_flat[-1] != 0:
            tour_flat.append(0)
            
    return tour_flat


def get_solution_costs(demands, n_vehicles, manager, routing, solution, distancesC):
    dist_ls = []
    load_ls = []
    tours = []
    costs = []
    for vehicle_id in range(n_vehicles):
        index = routing.Start(vehicle_id)
        route_distance = 0
        route_load = 0
        
        # Start node
        start_node = manager.IndexToNode(index)
        route = [start_node]
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += demands[node_index]
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(
                previous_index, index, vehicle_id
            )
            # Append next NODE index
            next_node = manager.IndexToNode(index)
            route.append(next_node)
            
        dist_ls.append(route_distance)
        load_ls.append(route_load)
        tours.append(route)
        costs.append(route_distance)
    return tours, costs, dist_ls, load_ls