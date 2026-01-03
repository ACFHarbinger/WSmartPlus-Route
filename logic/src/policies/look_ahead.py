import numpy as np
import pandas as pd
import gurobipy as gp

from gurobipy import GRB, quicksum
from typing import List

from logic.src.pipeline.simulator.processor import create_dataframe_from_matrix, convert_to_dict
from .look_ahead_aux import (
    should_bin_be_collected, add_bins_to_collect,
    compute_initial_solution, improved_simulated_annealing, 
    get_next_collection_day, update_fill_levels_after_first_collection,
)
from .hybrid_genetic_search import run_hgs
from .branch_cut_and_price import run_bcp
from .adaptive_large_neighborhood_search import run_alns, run_alns_package, run_alns_ortools


def policy_lookahead(
    binsids: List[int], 
    current_fill_levels: np.ndarray, 
    accumulation_rates: np.ndarray,  
    current_collection_day: int
  ):
  must_go_bins = []
  for i in binsids:
    if should_bin_be_collected(current_fill_levels[i], accumulation_rates[i]):
      must_go_bins.append(i)
  if must_go_bins != []:
    current_fill_levels = update_fill_levels_after_first_collection(binsids, must_go_bins, current_fill_levels)
    next_collection_day = get_next_collection_day(must_go_bins, current_fill_levels, accumulation_rates, binsids)
    must_go_bins = add_bins_to_collect(binsids, next_collection_day, must_go_bins, current_fill_levels, accumulation_rates, current_collection_day)
  return must_go_bins 


def policy_lookahead_vrpp(current_fill_levels, binsids, must_go_bins, distance_matrix, values, number_vehicles=8, env=None, time_limit=600):
    if number_vehicles == 0:
        number_vehicles = len(binsids)
    binsids = [0] + [bin_id + 1 for bin_id in binsids]
    
    must_go_bins = [must_go + 1 for must_go in must_go_bins]
    
    R = values.get('R')
    C = values.get('C') # expenses
    Omega = values.get('Omega', 0.1) 
    delta = values.get('delta', 0)   
    vehicle_capacity = values.get('vehicle_capacity')

    B = values.get('B', 19.0) 
    bin_volume = 2.5 
    
    percent_to_kg = (B * bin_volume) / 100.0
    R_model = R * percent_to_kg
    
    model = gp.Model("VRPP", env=env)
    model.Params.LogToConsole = 1
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0

    real_nodes = binsids[1:] 
    depot = 0
    fictitious_node = max(binsids) + 1
    all_nodes = binsids + [fictitious_node]

    # Variables
    x = model.addVars(all_nodes, all_nodes, vtype=GRB.BINARY, name="x")
    g = model.addVars(real_nodes, vtype=GRB.BINARY, name="g")
    f = model.addVars(all_nodes, all_nodes, vtype=GRB.CONTINUOUS, name="f")
    u = model.addVars(all_nodes, vtype=GRB.CONTINUOUS, name="u")

    
    capped_fills = {}
    for i in real_nodes:
        fill_val = current_fill_levels[i-1]
        if fill_val > 100:
            fill_val = 100 
        capped_fills[i] = fill_val

    obj_profit = quicksum(capped_fills[i] * g[i] for i in real_nodes) * R_model
    
    def get_dist(i, j):
        ii = 0 if i == fictitious_node else i
        jj = 0 if j == fictitious_node else j
        return distance_matrix[ii][jj]

    obj_cost = quicksum(x[i,j] * get_dist(i,j) for i in all_nodes for j in all_nodes if i != j) * C
    
    obj_penalty = quicksum(x[0, j] for j in real_nodes) * Omega 
    
    obj_must_go_reward = 0
    if must_go_bins:
        planned = [b for b in must_go_bins if b in real_nodes]
        if planned:
            M_reward = 100000.0 
            obj_must_go_reward = quicksum(g[i] for i in planned) * M_reward

    visit_bonus_val = 1.0 * R_model
    obj_visit_bonus = quicksum(g[i] for i in real_nodes) * visit_bonus_val

    model.setObjective(obj_profit + obj_must_go_reward + obj_visit_bonus - obj_cost - obj_penalty, GRB.MAXIMIZE)

    for j in real_nodes:
        model.addConstr(quicksum(x[i,j] for i in all_nodes if i != fictitious_node and i != j) == g[j])
    
    for j in real_nodes:
        model.addConstr(
            quicksum(x[i,j] for i in all_nodes if i != j) == 
            quicksum(x[j,k] for k in all_nodes if k != j)
        )
        
    model.addConstr(
        quicksum(x[0,j] for j in real_nodes) == 
        quicksum(x[i, fictitious_node] for i in real_nodes)
    )
    
    Q_val = vehicle_capacity / percent_to_kg
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                model.addConstr(f[i,j] <= Q_val * x[i,j])
    
    for j in real_nodes:
        model.addConstr(
            quicksum(f[i,j] for i in all_nodes if i!=j) + capped_fills[j]*g[j] == 
            quicksum(f[j,k] for k in all_nodes if k!=j)
        )
        
    for j in real_nodes:
        model.addConstr(f[0,j] == 0)

    N_count = len(all_nodes)
    for i in real_nodes:
        for j in real_nodes:
            if i != j:
                model.addConstr(u[i] - u[j] + N_count * x[i,j] <= N_count - 1)
    
    for i in real_nodes:
        model.addConstr(u[i] >= 1)
        model.addConstr(u[i] <= N_count)

    model.addConstr(quicksum(x[0,j] for j in real_nodes) <= number_vehicles)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        vals = model.getAttr('x', x)
        succ = {}
        for i in all_nodes:
            for j in all_nodes:
                if i!=j and vals[i,j] > 0.5:
                    succ[i] = j
        
        routes = []
        starts = [j for j in real_nodes if vals[0,j] > 0.5]
        for s in starts:
            route = [0]
            curr = s
            visited_count = 0
            while curr != fictitious_node and visited_count < len(real_nodes) + 2:
                route.append(curr)
                if curr not in succ:
                    break 
                curr = succ[curr]
                visited_count += 1
            routes.append(route)
        
        final_route = []
        for r in routes:
            final_route.extend(r)
            final_route.append(0)
            
        cleaned_route = []
        for n in final_route:
            if not cleaned_route or cleaned_route[-1] != n or n != 0:
                cleaned_route.append(n)
        if cleaned_route and cleaned_route[-1] != 0:
            cleaned_route.append(0)
            
        profit_val = model.ObjVal + obj_cost.getValue() + obj_penalty.getValue() 
        cost_val = obj_cost.getValue()
        
        return cleaned_route, profit_val, cost_val

    return [0, 0], 0, 0


def policy_lookahead_sans(data, bins_coordinates, distance_matrix, params, must_go_bins, values):
    T_init, iterations_per_T, alpha, T_min, *_ = params

    density, V, vehicle_capacity = values['B'], values['E'], values['vehicle_capacity'] 
    R, C, Omega = values['R'], values['C'], values['Omega'] 
    E, B, time_limit = 1, 1, values['time_limit']

    iframe = isinstance(data, pd.DataFrame)
    if iframe:
        pass
    else:
        data = create_dataframe_from_matrix(data)
        data['#bin'] = data['#bin'] + 1 

    coordinates_dict = convert_to_dict(bins_coordinates)
    
    id_to_index = {i: i for i in range(len(distance_matrix))}

    initial_routes = compute_initial_solution(data, coordinates_dict, distance_matrix, vehicle_capacity, id_to_index)

    must_go_bins = [b + 1 for b in must_go_bins]
    
    id_to_index = {i: i for i in range(len(distance_matrix))}

    current_route = []
    if initial_routes:
        current_route = initial_routes[0]
    else:
        current_route = [0, 0] 

    all_bins_set = set(data['#bin'].tolist()) - {0}
    route_bins_set = set(current_route) - {0}
    removed_bins = list(all_bins_set - route_bins_set)

    must_go_set = set(must_go_bins)
    missing_must_go = must_go_set - route_bins_set
    if missing_must_go:
        for b in missing_must_go:
            current_route.insert(1, b)
            if b in removed_bins:
                removed_bins.remove(b)

    single_route_solution = [current_route]

    optimized_routes, best_profit, last_distance, total_kg, last_revenue = improved_simulated_annealing(
        single_route_solution, distance_matrix, time_limit, id_to_index, data, vehicle_capacity,
        T_init, T_min, alpha, iterations_per_T, R, V, density, C, must_go_bins, removed_bins=set(removed_bins),
        perc_bins_can_overflow=values.get('perc_bins_can_overflow', 0.0), volume=V, density_val=density, max_vehicles=1
    )

    return optimized_routes, best_profit, last_distance


def policy_lookahead_hgs(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords):
    """
    Hybrid Genetic Search Policy.
    """
    B, E, Q = values['B'], values['E'], values['vehicle_capacity']
    R, C = values['R'], values['C']
    
    candidate_indices = [
        i for i, b_id in enumerate(binsids) 
        if current_fill_levels[i] > 0 or b_id in must_go_bins
    ]
    
    local_to_global = {local_idx: global_idx + 1 for local_idx, global_idx in enumerate(candidate_indices)} 
    
    demands = {}
    for local_i, global_i in local_to_global.items():
        bin_array_idx = global_i - 1 
        fill = current_fill_levels[bin_array_idx]
        weight = (fill / 100.0) * B * E
        demands[global_i] = weight
        
    global_must_go = set()
    binsids_map = {bid: i for i, bid in enumerate(binsids)}
    for mg in must_go_bins:
        if mg in binsids_map:
            global_must_go.add(binsids_map[mg] + 1)

    if not candidate_indices:
        return [0], 0, 0

    matrix_indices = list(local_to_global.values()) 
    
    bin_col = [0] + matrix_indices
    stock_col = [0.0]
    for idx in matrix_indices:
        val = demands.get(idx, 0)
        if idx in global_must_go:
            if val <= 0:
                val = 0.00001
        stock_col.append(val)
        
    accum_col = [0.0] * len(bin_col)
    
    seed_df = pd.DataFrame({
        '#bin': bin_col,
        'Stock': stock_col,
        'Accum_Rate': accum_col
    })
    
    coord_dict_seed = {}
    for idx in bin_col:
        if idx < len(coords):
            lat = coords.iloc[idx]['Lat']
            lng = coords.iloc[idx]['Lng']
            coord_dict_seed[idx] = (lat, lng)
        else:
            coord_dict_seed[idx] = (0,0)
            
    id_to_index_seed = { idx: idx for idx in bin_col }
    
    vrpp_routes = compute_initial_solution(
        seed_df, 
        coord_dict_seed, 
        distance_matrix, 
        Q, 
        id_to_index_seed
    )
    
    vrpp_tour_global = []
    if vrpp_routes:
        for route in vrpp_routes:
            vrpp_tour_global.extend(route)
            
    missing = [idx for idx in matrix_indices if idx not in vrpp_tour_global]
    vrpp_tour_global.extend(missing)
    vrpp_tour_global = [node for node in vrpp_tour_global if node != 0]    
    
    routes, fitness, cost = run_hgs(
        distance_matrix, 
        demands, 
        Q, 
        R, 
        C, 
        values, 
        global_must_go, 
        local_to_global, 
        vrpp_tour_global
    )
    
    
    final_sequence = []
    if routes:
        for route in routes:
            final_sequence.extend(route)
            final_sequence.append(0) 
        
    collected_bins_indices_tour = []
    for idx in final_sequence:
        if idx == 0:
            collected_bins_indices_tour.append(0) 
        else:
            collected_bins_indices_tour.append(idx)
             
    if not collected_bins_indices_tour:
        return [0, 0], 0, 0
    
    if collected_bins_indices_tour[0] != 0:
        collected_bins_indices_tour.insert(0, 0)
        
    if collected_bins_indices_tour[-1] != 0:
        collected_bins_indices_tour.append(0)
    
    return collected_bins_indices_tour, fitness, cost


def policy_lookahead_alns(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords, variant='custom'):
    """
    Adaptive Large Neighborhood Search Policy (Unified Dispatcher).
    """
    B, E, Q = values['B'], values['E'], values['vehicle_capacity']
    R, C = values['R'], values['C']
    
    candidate_indices = [
        i for i, b_id in enumerate(binsids) 
        if current_fill_levels[i] > 0 or b_id in must_go_bins
    ]
    
    local_to_global = {local_idx: global_idx + 1 for local_idx, global_idx in enumerate(candidate_indices)} 
    
    demands = {}
    for local_i, global_i in local_to_global.items():
        bin_array_idx = global_i - 1 
        fill = current_fill_levels[bin_array_idx]
        weight = (fill / 100.0) * B * E
        demands[global_i] = weight
        
    matrix_indices = list(local_to_global.values())
    if not matrix_indices:
        return [0, 0], 0, 0
        
    if variant == 'package':
        routes, cost = run_alns_package(distance_matrix, demands, Q, R, C, values)
    elif variant == 'ortools':
        routes, cost = run_alns_ortools(distance_matrix, demands, Q, R, C, values)
    else:
        routes, cost = run_alns(distance_matrix, demands, Q, R, C, values)
    
    final_sequence = [0]
    for route in routes:
        final_sequence.extend(route)
        final_sequence.append(0)
        
    return final_sequence, 0, cost


def policy_lookahead_bcp(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords, env=None):
    """
    Branch-Cut-and-Price Policy.
    """
    B, E, Q = values['B'], values['E'], values['vehicle_capacity']
    R, C = values['R'], values['C']
    
    candidate_indices = [
        i for i, b_id in enumerate(binsids) 
        if current_fill_levels[i] > 0 or b_id in must_go_bins
    ]
    
    local_to_global = {local_idx: global_idx + 1 for local_idx, global_idx in enumerate(candidate_indices)} 
    demands = {}
    
    binsids_map = {bid: i for i, bid in enumerate(binsids)}
    global_must_go = set()
    for mg in must_go_bins:
        if mg in binsids_map:
            global_must_go.add(binsids_map[mg] + 1)

    for local_i, global_i in local_to_global.items():
        bin_array_idx = global_i - 1 
        fill = current_fill_levels[bin_array_idx]
        weight = (fill / 100.0) * B * E
        demands[global_i] = weight
        
    matrix_indices = list(local_to_global.values())
    if not matrix_indices:
        return [0, 0], 0, 0
        
    routes, cost = run_bcp(distance_matrix, demands, Q, R, C, values, must_go_indices=global_must_go, env=None)
    
    final_sequence = [0]
    for route in routes:
        final_sequence.extend(route)
        final_sequence.append(0)
        
    return final_sequence, 0, cost