import numpy as np
import pandas as pd
import gurobipy as gp
import random
import time

from gurobipy import GRB, quicksum
from typing import List
from numpy.typing import NDArray
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
    current_fill_levels: NDArray[np.float64], 
    accumulation_rates: NDArray[np.float64], 
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
  return must_go_bins # bins that are mandatory to collect at the current day


def policy_lookahead_vrpp(current_fill_levels, binsids, must_go_bins, distance_matrix, values, number_vehicles=8, env=None):
    binsids = [0] + [bin_id + 1 for bin_id in binsids]
    must_go_bins = [must_go + 1 for must_go in must_go_bins]
    criticos = [bin_id in must_go_bins for bin_id in binsids]

    B, V, Q = values['B'], values['E'], values['vehicle_capacity'] #densidade, volume, capacidade
    R, C, Omega = values['R'], values['C'], 0.1 #receita, custo, omega
    delta, psi = 0, 1 #delta, psi

    enchimentos = np.insert(current_fill_levels, 0, 0.0)
    pesos_reais = [(e / 100) * B * V for e in enchimentos]
    nodes = list(range(len(binsids)))
    idx_deposito = 0
    nodes_real = [i for i in nodes if i != idx_deposito]
    S_dict = {i: pesos_reais[i] for i in nodes}
    criticos_dict = {i: criticos[i] for i in nodes}

    max_dist = 6000
    pares_viaveis = [(i, j) for i in nodes for j in nodes if i != j and distance_matrix[i][j] <= max_dist]
    mdl = gp.Model("VRPP", env=env) if env else gp.Model("VRPP")

    x = mdl.addVars(pares_viaveis, vtype=GRB.BINARY, name="x") #diz se a gente usa ou não a estrada que vai do ponto i até o ponto j
    y = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="y") # quanto de resíduo (kg) a gente está carregando nesse trecho entre i e j
    f = mdl.addVars(pares_viaveis, vtype=GRB.CONTINUOUS, lb=0, name="f") # pra evitar que o modelo crie "ciclos pequenos" fora do caminho principal (subtours).
    g = mdl.addVars(nodes, vtype=GRB.BINARY, name="g")
    k_var = mdl.addVar(lb=0, vtype=GRB.INTEGER, name="k_var")
    for i, j in pares_viaveis:
        mdl.addConstr(y[i, j] <= Q * x[i, j]) # limita que o trecho não tenha a capaciade maxima do caminhão
        mdl.addConstr(f[i, j] <= len(nodes) * x[i, j]) #evita subtours

    # Garante que o fluxo líquido em cada nó é igual ao resíduo gerado somente se o contentor for coletado
    for i in nodes_real:
        mdl.addConstr(quicksum(y[i, j] - y[j, i] for j in nodes if (i, j) in y or (j, i) in y) == S_dict[i] * g[i])

    # Teste de fixar o valor da quantidade de caminhões
    if number_vehicles == 0:
        number_vehicles = len(binsids)

    MAX_TRUCKS = number_vehicles
    mdl.addConstr(k_var <= MAX_TRUCKS)

    # Relaciona k_var com o número de rotas que partem e voltam do depósito.
    mdl.addConstr(k_var == quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x))
    mdl.addConstr(quicksum(x[idx_deposito, j] for j in nodes_real if (idx_deposito, j) in x) == k_var)
    mdl.addConstr(quicksum(x[j, idx_deposito] for j in nodes_real if (j, idx_deposito) in x) == k_var)

    # Se um contentor não for coletado (g[j]==0), não pode haver rota conectando-o ao depósito.
    for j in nodes_real:
        if (idx_deposito, j) in x: mdl.addConstr(x[idx_deposito, j] <= g[j])
        if (j, idx_deposito) in x: mdl.addConstr(x[j, idx_deposito] <= g[j])

    #Garante que pelo menos um número mínimo de contentores críticos seja visitado (ajustável por delta)
    mdl.addConstr(quicksum(g[i] for i in nodes_real if criticos_dict[i]) >= len([i for i in nodes_real if criticos_dict[i]]) - len(nodes_real) * delta)

    for i in nodes_real:
        # Esses contentores devem ser coletados.
        if criticos_dict[i] or enchimentos[i] >= psi * 100:
            mdl.addConstr(g[i] == 1)
        # g[i] é forçado a ser 0 (não coleta).
        if enchimentos[i] < 10 and not criticos[i]:
            g[i].ub = 0

    # Se um contentor for visitado (g[j]==1), deve ter exatamente uma entrada e uma saída.
    for j in nodes_real:
        mdl.addConstr(quicksum(x[i, j] for i in nodes if (i, j) in x) == g[j])
        mdl.addConstr(quicksum(x[j, k] for k in nodes if (j, k) in x) == g[j])

    # Assegura conectividade entre os pontos e impede a criação de ciclos menores isolados (subtours).
    mdl.addConstr(quicksum(f[0, j] for j in nodes_real if (0, j) in f) == quicksum(g[j] for j in nodes_real))
    for j in nodes_real:
        mdl.addConstr(quicksum(f[i, j] for i in nodes if (i, j) in f) - quicksum(f[j, k] for k in nodes if (j, k) in f) == g[j])

    mdl.setObjective(
        R * quicksum(S_dict[i] * g[i] for i in nodes_real)
        - 0.5 * C * quicksum(x[i, j] * distance_matrix[i][j] for i, j in pares_viaveis)
        - Omega * k_var,
        GRB.MAXIMIZE
    )

    mdl.Params.MIPFocus = 1
    mdl.Params.Heuristics = 0.5
    mdl.Params.Threads = 0
    mdl.Params.Cuts = 3
    mdl.Params.CliqueCuts = 2    # força clique cuts
    mdl.Params.CoverCuts = 2     # força cuts de conjuntos
    mdl.Params.FlowCoverCuts = 2 # força cortes para fluxos
    mdl.Params.GUBCoverCuts = 2
    mdl.Params.Presolve = 1
    mdl.Params.NodefileStart = 0.5
    mdl.setParam("MIPGap", 0.01)
    mdl.Params.TimeLimit = values['time_limit']

    contentores_coletados = []
    profit = 0
    cost = 0
    mdl.optimize()
    if mdl.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
        resultados_y = []
        resultados_g = []
        id_map = {i: binsids[i] for i in nodes}
        arcos_ativos = [(i, j) for i in nodes for j in nodes if i != j and x[i, j].X > 0.5]
        final_gap = mdl.MIPGap

        rotas = []
        visitados = set()
        while True:
            rota = []
            atual = 0
            while True:
                prox = [j for (i, j) in arcos_ativos if i == atual and (i, j) not in visitados]
                if not prox:
                    break
                j = prox[0]
                visitados.add((atual, j))
                rota.append((atual, j))
                atual = j
                if j == 0:
                    break
            if rota:
                rotas.append(rota)
            else:
                break

        for i in nodes:
            for j in nodes:
                if i != j and y[i, j].X > 0:
                    resultados_y.append((id_map[i], id_map[j], y[i, j].X))

        # Variáveis g[i]
        for i in nodes:
            if g[i].X > 0.5:
                resultados_g.append((id_map[i], g[i].X))

        for idx, rota in enumerate(rotas, start=1):
            df_rota = pd.DataFrame(
                [(id_map[i], id_map[j], x[i, j].X) for (i, j) in rota],
                columns=['i', 'j', 'x_ij']
            )
            
            contentores_coletados.extend([id_map[j] for (i, j) in rota])

        profit = (
            R * sum(S_dict[i] * g[i].X for i in nodes_real)
            - sum(x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis)
        )
            
        cost = sum(x[i, j].X * distance_matrix[i][j] for i, j in pares_viaveis)
        contentores_coletados = [contentor for contentor in contentores_coletados]
    return [0] + contentores_coletados, profit, cost


def policy_lookahead_sans(data, bins_coordinates, distance_matrix, params, must_go_bins, values):
    T_init, iterations_per_T, alpha, T_min, *_ = params

    density, V, vehicle_capacity = values['B'], values['E'], values['vehicle_capacity'] # densidade, volume, capacidade
    R, C, Omega = values['R'], values['C'], 0.1 # receita, custo, omega
    E, B, time_limit = 1, 1, values['time_limit']

    iframe = isinstance(data, pd.DataFrame)
    if iframe:
        # Data passed from simulator (101 rows, 0=Depot, 1..100=Bins)
        # Indexes are already 1-based (Depot=0). No shift needed.
        pass
    else:
        # Legacy matrix input (0-based)
        data = create_dataframe_from_matrix(data)
        data['#bin'] = data['#bin'] + 1 # Shift 0..99 -> 1..100

    coordinates_dict = convert_to_dict(bins_coordinates)
    
    # Map IDs (1..100) -> Matrix Indices (1..100).
    # Since Distance Matrix has Depot at 0, Bin ID k matches Matrix Index k.
    id_to_index = {i: i for i in range(len(distance_matrix))}

    # 1. Compute greedy multi-route solution
    # compute_initial_solution expects 'data' to have '#bin' values that are keys in id_to_index.
    initial_routes = compute_initial_solution(data, coordinates_dict, distance_matrix, vehicle_capacity, id_to_index)

    # 2. Update Must Go Bins (Input 0..99 -> Shift to 1..100)
    must_go_bins = [b + 1 for b in must_go_bins]
    
    # Update ID map (1 -> 1, ... 100 -> 100) if we assume matrix aligns
    # Before: 0->0, ..., 99->99.
    # Now we use nodes 1..100. distance_matrix[1] corresponds to Bin 0 (which is now Node 1).
    # So mapping Node k -> Matrix Index k is correct.
    id_to_index = {i: i for i in range(len(distance_matrix))}

    # 2. ADAPT FOR SINGLE VEHICLE SIMULATION (n_vehicles=1)
    # The simulator (day.py) only executes routes[0]. We must focus optimization on this single route.
    # Strategy:
    #   - Take initial_routes[0] as current_route.
    #   - Enforce inclusion of ALL must_go_bins into current_route (even if it temporarily overloads capacity).
    #   - Move all other bins (from initial_routes[1:] or unvisited) into 'removed_bins'.
    #   - SANS will then optimize by swapping non-mandatory bins between current_route and removed_bins.

    current_route = []
    if initial_routes:
        current_route = initial_routes[0]
    else:
        current_route = [0, 0] # Depot-only if empty

    # Identify all bins in the system
    all_bins_set = set(data['#bin'].tolist()) - {0}

    # Identify bins currently in Route 1
    route_bins_set = set(current_route) - {0}

    # Prepare removed_bins (starts with everything NOT in Route 1)
    removed_bins = list(all_bins_set - route_bins_set)

    # Force insert missing must_go_bins into current_route
    # We remove them from 'removed_bins' and append to current_route
    must_go_set = set(must_go_bins)
    missing_must_go = must_go_set - route_bins_set
    if missing_must_go:
        # Insert them at a reasonable position (e.g., end before depot, or index 1)
        # Simple approach: Insert at index 1
        for b in missing_must_go:
            current_route.insert(1, b)
            if b in removed_bins:
                removed_bins.remove(b)

    # Re-package as the solution for SANS
    # SANS expects a list of routes. We give it [current_route].
    single_route_solution = [current_route]

    optimized_routes, best_profit, last_distance, total_kg, last_revenue = improved_simulated_annealing(
        single_route_solution, distance_matrix, time_limit, id_to_index, data, vehicle_capacity,
        T_init, T_min, alpha, iterations_per_T, R, V, density, C, must_go_bins, removed_bins=set(removed_bins),
        perc_bins_can_overflow=values.get('perc_bins_can_overflow', 0.0), volume=V, density_val=density, max_vehicles=1
    )

    # Output is already 1-based, no shift needed
    return optimized_routes, best_profit, last_distance


def policy_lookahead_hgs(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords):
    """
    Hybrid Genetic Search Policy.
    """
    # 1. Parse Parameters
    B, E, Q = values['B'], values['E'], values['vehicle_capacity']
    R, C = values['R'], values['C']
    
    # 2. Filter nodes: We consider all bins passed in 'binsids' as candidates,
    # but the HGS will decide order. 
    # Note: If you want to ONLY route specific bins, filter 'binsids' here.
    # The Gurobi model has a choice (g variable). HGS usually routes everyone in the list.
    # To mimic Gurobi's 'selection', we can include all bins with fill > 0.
    
    candidate_indices = [
        i for i, b_id in enumerate(binsids) 
        if current_fill_levels[i] > 0 or b_id in must_go_bins
    ]
    
    # Create mapping
    # HGS works on indices 0..N. We map these to the distance matrix indices.
    # The distance matrix includes depot at 0. 'binsids' usually start from index 1 in matrix?
    # Assuming binsids[i] corresponds to row/col i+1 in distance_matrix (since 0 is depot).
    
    # Map local HGS index -> Matrix/Bin Index
    local_to_global = {local_idx: global_idx + 1 for local_idx, global_idx in enumerate(candidate_indices)} 
    
    # Prepare Demands (Weights)
    demands = {}
    for local_i, global_i in local_to_global.items():
        # global_i is matrix index. binsids index is global_i - 1
        bin_array_idx = global_i - 1 
        fill = current_fill_levels[bin_array_idx]
        weight = (fill / 100.0) * B * E
        demands[global_i] = weight
        
    # Map must_go_bins (IDs) to Global Matrix Indices
    # We need this for both Seeding (epsilon stock) and Evaluation (penalties)
    global_must_go = set()
    # Create valid ID lookup
    binsids_map = {bid: i for i, bid in enumerate(binsids)}
    for mg in must_go_bins:
        if mg in binsids_map:
            # binsids index i correponds to Matrix Index i + 1
            global_must_go.add(binsids_map[mg] + 1)

    if not candidate_indices:
        print(f"DEBUG: No candidates found. Fill > 0 count: {sum(current_fill_levels > 0)}")
        return [0], 0, 0

    # --- SEEDING WITH VRPP (GREEDY) SOLUTION ---
    # Robust approach: Use Matrix Indices (0..N) as IDs, mirroring SANS structure.
    # This avoids type mismatches (String vs Int) and ordering assumptions in solutions.py.
    
    # Matrix Indices involved: 0 (Depot) and those in local_to_global (Bins)
    matrix_indices = list(local_to_global.values()) # e.g. [1, 2, ... 100]
    
    # Construct columns for DataFrame
    # Row 0 is Depot
    bin_col = [0] + matrix_indices
    
    # Stock: Depot=0. Bins=demands[idx]
    stock_col = [0.0]
    for idx in matrix_indices:
        val = demands.get(idx, 0)
        # Force VRPP to visit must_go_bins even if empty (epsilon stock)
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
    
    # Coords Dict: Map Matrix Index -> (Lat, Lon)
    # Assumes coords list is aligned with Matrix Indices
    coord_dict_seed = {}
    for idx in bin_col:
        # idx is the Matrix Index (0..N).
        # We assume coords DataFrame has at least N+1 rows, ordered such that row `idx` corresponds to Node `idx`.
        # setup_df in processor.py ensures this ordering/reset_index.
        if idx < len(coords):
            # Access by position using iloc
            lat = coords.iloc[idx]['Lat']
            lng = coords.iloc[idx]['Lng']
            coord_dict_seed[idx] = (lat, lng)
        else:
            coord_dict_seed[idx] = (0,0)
            
    # Identity ID Map
    id_to_index_seed = { idx: idx for idx in bin_col }
    
    # Run VRPP Greedy
    vrpp_routes = compute_initial_solution(
        seed_df, 
        coord_dict_seed, 
        distance_matrix, 
        Q, 
        id_to_index_seed
    )
    
    # Result is list of lists of Matrix Indices (since we passed Matrix Indices as IDs).
    vrpp_tour_global = []
    if vrpp_routes:
        for route in vrpp_routes:
            vrpp_tour_global.extend(route)
            
    # Ensure all required nodes are present
    missing = [idx for idx in matrix_indices if idx not in vrpp_tour_global]
    vrpp_tour_global.extend(missing)
    
    # Remove Depot (0) if present, as HGS works on Bin permutations
    vrpp_tour_global = [node for node in vrpp_tour_global if node != 0]
    
    # -------------------------------------------

    # 3. Run HGS Dispatcher
    # Pass all necessary data
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
    
    # 5. Format Output
    # Convert routes to flat list of IDs (excluding depot 0 inside the list, 
    # but the simulator usually expects [0, id, id, 0, id...]).
    # The 'policy_lookahead_vrpp' returns [0] + contentores_coletados.
    
    final_sequence = []
    # Support Multi-Route (Concatenate routes with returns to depot)
    if routes:
        for route in routes:
            final_sequence.extend(route)
            final_sequence.append(0) # Return to depot after each route
        
        # Remove the very last 0 if we want to rely on the final append below?
        # The logic below returns [0] + IDs + [0].
        # If final_sequence is [1, 2, 0, 3, 4, 0], we want result [0, 1, 2, 0, 3, 4, 0].
        # So we should POP the last 0 if present to avoid double 0 at end with logic below?
        # Actually, let's just construct the full tour here cleanly.
        
    # Convert Matrix Indices (1..N) back to Bin IDs
    # The standard output seems to be the list of collected bin INDICES (0-based relative to binsids).
    
    collected_bins_indices_tour = []
    for idx in final_sequence:
        if idx == 0:
            collected_bins_indices_tour.append(0) # Depot
        else:
            # Return Matrix Index (1..N) directly, as simulator expects 1-based Bin IDs
            collected_bins_indices_tour.append(idx)
             
    # Ensure start and end with 0 if not empty
    if not collected_bins_indices_tour:
        return [0, 0], 0, 0
    
    # Check if starts with 0
    if collected_bins_indices_tour[0] != 0:
        collected_bins_indices_tour.insert(0, 0)
        
    # Check if ends with 0
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
        
    # Dispatch specific runner
    if variant == 'package':
        routes, cost = run_alns_package(distance_matrix, demands, Q, R, C, values)
    elif variant == 'ortools':
        routes, cost = run_alns_ortools(distance_matrix, demands, Q, R, C, values)
    else:
        routes, cost = run_alns(distance_matrix, demands, Q, R, C, values)
    
    # 3. Format Output
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
    
    # Map must_go_bins to Global Indices (for Penalty enforcement)
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