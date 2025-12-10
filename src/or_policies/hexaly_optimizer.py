import io
import sys
import numpy as np
import hexaly.optimizer as hx

from typing import List, Tuple
from numpy.typing import NDArray
from src.pipeline.simulator.loader import load_area_and_waste_type_params


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
        max_iter_no_improv: int=10
    ) -> Tuple[List[int], float, float]:
    """
    Vehicle Routing Problem with Profits using Hexaly Optimizer.
    Updated for Gurobi-like precision using Integer Scaling and Hard Constraints.
    Includes Early Stopping Callback.
    
    Returns:
        Tuple containing:
        - List of visited nodes (starting with depot 0)
        - Profit (float)
        - Cost (float)
    """
    
    # ---------------------------------------------------------
    # 0. PARAMETERS & SCALING
    # ---------------------------------------------------------
    # SCALING FACTOR: Convert floats to Ints to match MIP tightness
    SCALE = 1000 
    
    Omega, delta, psi = 0.1, 0, 1
    Q, R, B, C, V = load_area_and_waste_type_params(area, waste_type)
    
    # Scale Capacity to Int
    Q_int = int(Q * SCALE)

    # ---------------------------------------------------------
    # 1. DATA PREPARATION
    # ---------------------------------------------------------
    n_bins = len(bins)
    enchimentos = np.insert(bins, 0, 0.0) # Depot has 0 fill

    # Calculate Real Weights (Float) then Scale to Int
    pesos_reais_float = [(e / 100) * B * V for e in enchimentos]
    pesos_reais_int = [int(w * SCALE) for w in pesos_reais_float]

    # Scale Distance Matrix to Int
    dist_matrix_int = [
        [int(dist * SCALE) for dist in row] 
        for row in distancematrix
    ]

    nodes = list(range(n_bins + 1))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]
    
    # Determine must-go and mandatory containers
    must_go = set()
    for container_id in range(n_bins):
        pred_value = bins[container_id] + media[container_id] + param * desviopadrao[container_id]
        if pred_value >= 100:
            must_go.add(container_id + 1)
            
    mandatory = set()
    for i in nodes_real:
        if (i in must_go) or (enchimentos[i] >= psi * 100):
            mandatory.add(i)

    # Identify Forbidden Nodes (Low Fill & Not Critical - Matching Gurobi)
    forbidden = set()
    for i in nodes_real:
        if enchimentos[i] < 10 and not (i in must_go):
            forbidden.add(i)

    max_dist_int = 6000 * SCALE
    num_nodes = len(nodes)
    
    # ---------------------------------------------------------
    # 2. HEXALY MODEL
    # ---------------------------------------------------------
    with hx.HexalyOptimizer() as optimizer:
        model = optimizer.model
        
        # Pass Scaled Data to Model
        dist_array = model.array(dist_matrix_int)
        weights_array = model.array(pesos_reais_int)
        
        # Decision variables: Sequence of nodes for each vehicle
        routes = [model.list(num_nodes) for _ in range(number_vehicles)]
        
        # Constraint: Disjoint routes (each node visited at most once)
        model.constraint(model.disjoint(routes))
        
        # Define union of visited nodes
        all_visited = model.union(routes)
        model.constraint(model.contains(all_visited, idx_deposito) == 0)

        # Hard Forbidden Constraint (Matching Gurobi g[i].ub = 0)
        for node in forbidden:
            model.constraint(model.contains(all_visited, node) == 0)

        # Optimization Expressions
        total_profit_int = 0
        total_dist_int = 0
        vehicles_used_expr = 0
        
        for k in range(number_vehicles):
            route = routes[k]
            count = model.count(route)
            
            # 1. Vehicle Used?
            is_used = model.iif(count > 0, 1, 0)
            vehicles_used_expr += is_used
            
            # 2. Load Calculation (Integer)
            route_load = model.sum(
                model.range(0, count),
                model.lambda_function(lambda i: model.at(weights_array, model.at(route, i)))
            )
            
            # Capacity Constraint (Hard)
            model.constraint(route_load <= Q_int)
            
            # Profit Contribution (Sum of loads, scaled R applied later)
            total_profit_int += route_load 
            
            # 3. Distance Calculation (Integer)
            d_start = model.iif(
                count > 0,
                model.at(dist_array, idx_deposito, model.at(route, 0)),
                0
            )
            d_path = model.sum(
                model.range(0, count - 1),
                model.lambda_function(
                    lambda i: model.at(dist_array, model.at(route, i), model.at(route, i + 1))
                )
            )
            d_end = model.iif(
                count > 0,
                model.at(dist_array, model.at(route, count - 1), idx_deposito),
                0
            )
            
            route_dist = d_start + d_path + d_end
            total_dist_int += route_dist
            
            # Max Distance Constraints
            if max_dist_int > 0:
                model.constraint(route_dist <= max_dist_int)

        # Mandatory Visits
        for node in mandatory:
            model.constraint(model.contains(all_visited, node))
            
        # Must-Go Flexible Constraint
        must_go_list = list(must_go)
        if must_go_list:
            nb_must_go = len(must_go_list)
            visited_critical = model.sum(
                [model.contains(all_visited, node) for node in must_go_list]
            )
            limit = max(0, nb_must_go - int(len(nodes_real) * delta))
            model.constraint(visited_critical >= limit)

        # ---------------------------------------------------------
        # 3. OBJECTIVE FUNCTION (Implicit Type Promotion)
        # ---------------------------------------------------------
        # Rely on Hexaly's expression system to handle implicit float promotion
        # when dividing the HxExpression (int) by the Python constant SCALE.
        
        revenue_term = (total_profit_int / SCALE) * R 
        dist_cost_term = (total_dist_int / SCALE) * (0.5 * C) 
        vehicle_cost_term = vehicles_used_expr * Omega
        
        obj = revenue_term - dist_cost_term - vehicle_cost_term
        
        model.maximize(obj)
        model.close()
        
        # ---------------------------------------------------------
        # 4. SOLVE WITH CALLBACK
        # ---------------------------------------------------------
        optimizer.param.time_limit = time_limit
        optimizer.param.verbosity = 0
        
        # --- EARLY STOPPING CALLBACK ---
        best_obj_val = [None]
        no_improv_iter = [0]
        
        def callback(opt, type):
            # Check feasibility first
            if opt.solution.status == hx.HxSolutionStatus.INFEASIBLE:
                return
            
            if type == hx.HxCallbackType.TIME_TICKED:
                # Get current objective value
                val = opt.solution.get_value(obj)
                
                if best_obj_val[0] is None:
                    best_obj_val[0] = val
                    no_improv_iter[0] = 0
                else:
                    # Check for significant relative improvement (e.g. 0.1%)
                    # Use max(..., 1e-10) to avoid division by zero
                    relative_improvement = abs(val - best_obj_val[0]) / max(abs(best_obj_val[0]), 1e-10)
                    
                    if relative_improvement > 0.001:
                        best_obj_val[0] = val
                        no_improv_iter[0] = 0
                    else:
                        no_improv_iter[0] += 1
                    
                    if no_improv_iter[0] >= max_iter_no_improv:
                        # Stop the optimizer if no improvement for 'max_iter_no_improv' ticks
                        opt.stop()

        # Register the callback
        optimizer.add_callback(hx.HxCallbackType.TIME_TICKED, callback)
        # -------------------------------

        # Output capture
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        
        try:
            optimizer.solve()
            sys.stdout = old_stdout
            
            # ---------------------------------------------------------
            # 5. POST-PROCESSING (Unscale)
            # ---------------------------------------------------------
            final_route_flat = [0]
            
            calc_revenue = 0.0
            calc_dist = 0.0
            
            for k in range(number_vehicles):
                r_vals = list(routes[k].value)
                if not r_vals:
                    continue
                    
                final_route_flat.extend(r_vals)
                final_route_flat.append(0)
                
                # Re-calculate exact float values for return
                for node in r_vals:
                    calc_revenue += pesos_reais_float[node] * R
                
                # Distance
                if len(r_vals) > 0:
                    calc_dist += distancematrix[idx_deposito][r_vals[0]]
                    for i in range(len(r_vals) - 1):
                        calc_dist += distancematrix[r_vals[i]][r_vals[i+1]]
                    calc_dist += distancematrix[r_vals[-1]][idx_deposito]
            
            profit = calc_revenue - calc_dist
            cost = calc_dist
            return final_route_flat, profit, cost
        except Exception as e:
            sys.stdout = old_stdout
            print(f"Hexaly optimization failed: {e}")
            return [0], 0.0, 0.0