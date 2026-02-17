"""
OR-Tools engine for Branch-Cut-and-Price module.
"""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def run_bcp_ortools(
    dist_matrix: np.ndarray,
    demands: Dict[int, float],
    capacity: float,
    R: float,
    C: float,
    values: Dict[str, Any],
    mandatory_nodes: Optional[List[int]] = None,
) -> Tuple[List[List[int]], float]:
    """
    Solve Prize-Collecting CVRP using Google OR-Tools.

    Args:
        dist_matrix: NxN distance matrix.
        demands: Dictionary of node demands.
        capacity: Maximum vehicle capacity.
        R: Revenue multiplier.
        C: Cost multiplier.
        values: Config dictionary (time_limit, num_vehicles).
        mandatory_nodes: Optional list of mandatory node indices.

    Returns:
        Tuple[List[List[int]], float]: (routes, total_cost)
    """
    num_nodes = len(dist_matrix)
    num_vehicles = values.get("num_vehicles", num_nodes)  # Dynamic fleet
    depot_index = 0

    # 1. Create Routing Manager and Model
    manager = pywrapcp.RoutingIndexManager(num_nodes, num_vehicles, depot_index)
    routing = pywrapcp.RoutingModel(manager)

    # 2. Define Cost distance callback
    SCALE = 1000  # OR-Tools works better with integers

    def distance_callback(from_index, to_index):
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return int(dist_matrix[from_node][to_node] * SCALE)

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 3. Add Capacity Constraints
    def demand_callback(from_index):
        from_node = manager.IndexToNode(from_index)
        if from_node == 0:
            return 0
        return int(demands.get(from_node, 0))

    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

    routing.AddDimensionWithVehicleCapacity(demand_callback_index, 0, [int(capacity)] * num_vehicles, True, "Capacity")

    # 4. Add Penalties (Prize Collecting)
    MUST_GO_PENALTY = 1_000_000_000
    m_set = set(mandatory_nodes) if mandatory_nodes else set()

    for i in range(1, num_nodes):
        d = demands.get(i, 0)
        revenue = d * R

        penalty = MUST_GO_PENALTY if i in m_set else int(revenue * SCALE)

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

        real_dist_cost = 0.0
        for r in routes:
            full_r = [0] + r + [0]
            for i in range(len(full_r) - 1):
                u, v = full_r[i], full_r[i + 1]
                real_dist_cost += float(dist_matrix[u][v])

        return routes, real_dist_cost * C

    return [], 0.0
