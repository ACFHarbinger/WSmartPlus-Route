import io
import sys
import numpy as np
import hexaly.optimizer as hx

from typing import List
from numpy.typing import NDArray
from app.src.pipeline.simulator.loader import load_area_and_waste_type_params


def policy_hexaly_vrpp(
        bins: NDArray[np.float64],
        distancematrix: List[List[float]], 
        param: float, 
        media: NDArray[np.float64], 
        desviopadrao: NDArray[np.float64], 
        waste_type: str='plastic', 
        area: str='riomaior', 
        number_vehicles: int=1, 
        time_limit: int=60,
        max_iter_no_improv: int=3
    ):
    """
    Vehicle Routing Problem with Profits using Hexaly Optimizer.
    
    Args:
        bins: Array of bin fill levels (percentage)
        distancematrix: Distance matrix between nodes
        param: Parameter for standard deviation multiplier
        media: Mean fill level predictions
        desviopadrao: Standard deviation of predictions
        waste_type: Type of waste (e.g., 'plastic')
        area: Area name (e.g., 'riomaior')
        number_vehicles: Maximum number of vehicles
        time_limit: Time limit in seconds
        max_iter_no_improv: Maximum iterations without improving profit before stopping
    
    Returns:
        List of routes, where each route is a list of node indices
    """
    Omega, delta, psi = 0.1, 0, 1
    Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
    
    # Convert bin fill percentages to actual weights in KG
    pesos_reais = [(e / 100) * B * V for e in bins]
    nodes = list(range(len(bins)))
    idx_deposito = 0  # depot index
    nodes_real = [i for i in nodes if i != idx_deposito]
    
    # Determine must-go containers
    must_go = {}
    must_go_count = 0
    for container_id in range(len(bins)):
        pred_value = bins[container_id] + media[container_id] + param * desviopadrao[container_id]
        must_go[container_id] = pred_value >= 100
        if must_go[container_id] and container_id != idx_deposito:
            must_go_count += 1
    
    # Mandatory containers (critical or above threshold)
    mandatory = set()
    for i in nodes_real:
        if must_go[i] or bins[i] >= psi * 100:
            mandatory.add(i)
    
    max_dist = 6000
    num_nodes = len(nodes)
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model
        
        # Convert distance matrix and weights to Hexaly arrays
        dist_array = model.array(distancematrix)
        weights_array = model.array(pesos_reais)
        
        # Decision variables: sequence of visits for each vehicle
        routes = [model.list(num_nodes) for _ in range(number_vehicles)]
        
        # Constraint: each node appears at most once across all routes
        model.constraint(model.partition(routes))
        
        # Create distance and load calculations for each route
        route_distances = []
        route_loads = []
        route_profits = []
        for k in range(number_vehicles):
            route = routes[k]
            route_length = model.count(route)
            
            # Distance calculation: depot -> route -> depot
            # Distance from depot to first node
            dist_to_first = model.iif(
                route_length > 0,
                model.at(dist_array, idx_deposito, model.at(route, 0)),
                0
            )
            
            # Distances between consecutive nodes in route
            dist_between = model.sum(
                model.range(0, route_length - 1),
                model.lambda_function(
                    lambda i: model.at(dist_array, model.at(route, i), model.at(route, i + 1))
                )
            )
            
            # Distance from last node back to depot
            dist_to_depot = model.iif(
                route_length > 0,
                model.at(dist_array, model.at(route, route_length - 1), idx_deposito),
                0
            )
            
            # Total distance
            dist_expr = dist_to_first + dist_between + dist_to_depot
            route_distances.append(dist_expr)
            
            # Load calculation: sum of weights in route
            load_expr = model.sum(
                model.range(0, route_length),
                model.lambda_function(lambda i: model.at(weights_array, model.at(route, i)))
            )
            route_loads.append(load_expr)
            
            # Profit calculation: revenue from collected waste (same as load)
            route_profits.append(load_expr)
            
            # Capacity constraint
            model.constraint(load_expr <= Q)
            
            # Distance constraints are handled implicitly by max_dist filtering
            # We create a lambda to check each edge in the route
            model.constraint(
                model.sum(
                    model.range(0, route_length - 1),
                    model.lambda_function(
                        lambda i: model.iif(
                            model.at(dist_array, model.at(route, i), model.at(route, i + 1)) <= max_dist,
                            1,
                            0
                        )
                    )
                ) == route_length - 1
            )
            
            # Depot connection distance constraints
            model.constraint(
                model.iif(
                    route_length > 0,
                    model.and_(
                        model.at(dist_array, idx_deposito, model.at(route, 0)) <= max_dist,
                        model.at(dist_array, model.at(route, route_length - 1), idx_deposito) <= max_dist
                    ),
                    1  # True if route is empty
                )
            )
        
        # Mandatory containers must be visited
        all_visited = model.union(routes)
        for node in mandatory:
            model.constraint(model.contains(all_visited, node))
        
        # Must-go containers constraint (with delta flexibility)
        must_go_nodes = [i for i in nodes_real if must_go[i]]
        if must_go_nodes:
            min_must_go = max(0, len(must_go_nodes) - int(len(nodes_real) * delta))
            visited_must_go = model.sum(
                [model.contains(all_visited, node) for node in must_go_nodes]
            )
            model.constraint(visited_must_go >= min_must_go)
        
        # Objective: maximize profit - cost
        total_profit = R * model.sum(route_profits)
        total_distance_cost = 0.5 * C * model.sum(route_distances)
        num_vehicles_used = model.sum(
            [model.iif(model.count(routes[k]) > 0, 1, 0) for k in range(number_vehicles)]
        )
        vehicle_cost = Omega * num_vehicles_used
        
        objective = total_profit - total_distance_cost - vehicle_cost
        model.maximize(objective)
        
        model.close()
        
        # Set parameters
        optimizer.param.time_limit = time_limit
        optimizer.param.verbosity = 0
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()  # Suppress output
        
        # Solve
        best_solution = [None]
        no_improvement_count = [0]
        def callback(optimizer, callback_type):
            """Callback to check if solution is not improving"""
            if optimizer.solution.status == hx.HxSolutionStatus.INFEASIBLE:
                return
                
            if callback_type == hx.HxCallbackType.TIME_TICKED:
                current_value = optimizer.solution.get_value(objective)
                if best_solution[0] is None:
                    best_solution[0] = current_value
                    no_improvement_count[0] = 0
                else:
                    improvement = abs(current_value - best_solution[0])
                    relative_improvement = improvement / max(abs(best_solution[0]), 1e-10)
                    
                    if relative_improvement > 0.001:  # 0.1% improvement threshold
                        best_solution[0] = current_value
                        no_improvement_count[0] = 0
                    else:
                        no_improvement_count[0] += 1
                    
                    # Stop if no better solution is found for too long
                    if (no_improvement_count[0] >= max_iter_no_improv):
                        optimizer.stop()

        optimizer.add_callback(hx.HxCallbackType.TIME_TICKED, callback)
        try:
            optimizer.solve()
            sys.stdout = old_stdout  # Restore output

            # Extract solution
            solution_routes = []
            for k in range(number_vehicles):
                route_list = routes[k].value
                if len(route_list) > 0:
                    # Add depot at start and end
                    full_route = [idx_deposito] + list(route_list) + [idx_deposito]
                    solution_routes.append(full_route)
            
            return solution_routes
        except hx.HxError as e:
            print("Hexaly optimization failed. Error message:", str(e))
            return None
