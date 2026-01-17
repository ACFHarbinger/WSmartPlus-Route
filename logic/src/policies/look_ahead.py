"""
Look-ahead policy module.

This module implements policies that look ahead into the future to optimize
collection schedules compared to immediate greedy or periodic policies.
"""

from typing import Any, List, Tuple

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB, quicksum

from logic.src.pipeline.simulator.loader import load_area_and_waste_type_params
from logic.src.pipeline.simulator.processor import (
    convert_to_dict,
    create_dataframe_from_matrix,
)

from .adapters import IPolicy, PolicyRegistry
from .adaptive_large_neighborhood_search import (
    run_alns,
    run_alns_ortools,
    run_alns_package,
)
from .branch_cut_and_price import run_bcp
from .hybrid_genetic_search import run_hgs
from .lin_kernighan import solve_lk
from .look_ahead_aux import (
    add_bins_to_collect,
    compute_initial_solution,
    get_next_collection_day,
    improved_simulated_annealing,
    should_bin_be_collected,
    update_fill_levels_after_first_collection,
)
from .look_ahead_aux.routes import create_points
from .look_ahead_aux.solutions import find_solutions
from .single_vehicle import find_route, get_route_cost, local_search_2opt


def policy_lookahead(
    binsids: List[int],
    current_fill_levels: np.ndarray,
    accumulation_rates: np.ndarray,
    current_collection_day: int,
) -> List[int]:
    """
    Identify bins that must be collected today or in the near future.

    Args:
      binsids (List[int]): List of bin IDs.
      current_fill_levels (np.ndarray): Current fill levels (0-100%).
      accumulation_rates (np.ndarray): Daily waste accumulation rates.
      current_collection_day (int): The current simulation day.

    Returns:
      List[int]: List of bin IDs that are marked for collection.
    """
    must_go_bins = []
    for i in binsids:
        if should_bin_be_collected(current_fill_levels[i], accumulation_rates[i]):
            must_go_bins.append(i)
    if must_go_bins != []:
        current_fill_levels = update_fill_levels_after_first_collection(binsids, must_go_bins, current_fill_levels)
        next_collection_day = get_next_collection_day(must_go_bins, current_fill_levels, accumulation_rates, binsids)
        must_go_bins = add_bins_to_collect(
            binsids,
            next_collection_day,
            must_go_bins,
            current_fill_levels,
            accumulation_rates,
            current_collection_day,
        )
    return must_go_bins


def policy_lookahead_vrpp(
    current_fill_levels,
    binsids,
    must_go_bins,
    distance_matrix,
    values,
    number_vehicles=8,
    env=None,
    time_limit=600,
):
    """
    Look-ahead policy using a Gurobi-based VRPP optimizer.

    Predicts future overflows and optimizes the current day's collection to
    maximize profit and prevent overflows.

    Args:
      current_fill_levels: Current fill levels for all bins.
      binsids: List of all bin IDs.
      must_go_bins: Bins that must be visited (predicted to overflow).
      distance_matrix: NxN distance matrix.
      values: Problem parameters (R, C, vehicle_capacity, etc.).
      number_vehicles: Fleet size. Default: 8.
      env: Gurobi environment.
      time_limit: Optimizer time limit in seconds. Default: 600.

    Returns:
      Tuple[List[int], float, float]: Cleaned tour, profit, and travel cost.
    """
    if number_vehicles == 0:
        number_vehicles = len(binsids)
    binsids = [0] + [bin_id + 1 for bin_id in binsids]

    must_go_bins = [must_go + 1 for must_go in must_go_bins]

    R = values.get("R")
    C = values.get("C")  # expenses
    Omega = values.get("Omega", 0.1)
    # delta = values.get('delta', 0)
    vehicle_capacity = values.get("vehicle_capacity")

    B = values.get("B", 19.0)
    bin_volume = 2.5

    percent_to_kg = (B * bin_volume) / 100.0
    R_model = R * percent_to_kg

    model = gp.Model("VRPP", env=env)
    model.Params.LogToConsole = 1
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = 0.0

    real_nodes = binsids[1:]
    # depot = 0
    fictitious_node = max(binsids) + 1
    all_nodes = binsids + [fictitious_node]

    # Variables
    x = model.addVars(all_nodes, all_nodes, vtype=GRB.BINARY, name="x")
    g = model.addVars(real_nodes, vtype=GRB.BINARY, name="g")
    f = model.addVars(all_nodes, all_nodes, vtype=GRB.CONTINUOUS, name="f")
    u = model.addVars(all_nodes, vtype=GRB.CONTINUOUS, name="u")

    capped_fills = {}
    for i in real_nodes:
        fill_val = current_fill_levels[i - 1]
        if fill_val > 100:
            fill_val = 100
        capped_fills[i] = fill_val

    obj_profit = quicksum(capped_fills[i] * g[i] for i in real_nodes) * R_model

    def get_dist(i, j):
        """
        Helper to get distance between two nodes, handling the fictitious node.
        """
        ii = 0 if i == fictitious_node else i
        jj = 0 if j == fictitious_node else j
        return distance_matrix[ii][jj]

    obj_cost = quicksum(x[i, j] * get_dist(i, j) for i in all_nodes for j in all_nodes if i != j) * C

    obj_penalty = quicksum(x[0, j] for j in real_nodes) * Omega

    obj_must_go_reward = 0
    if must_go_bins:
        planned = [b for b in must_go_bins if b in real_nodes]
        if planned:
            M_reward = 100000.0
            obj_must_go_reward = quicksum(g[i] for i in planned) * M_reward

    visit_bonus_val = 1.0 * R_model
    obj_visit_bonus = quicksum(g[i] for i in real_nodes) * visit_bonus_val

    model.setObjective(
        obj_profit + obj_must_go_reward + obj_visit_bonus - obj_cost - obj_penalty,
        GRB.MAXIMIZE,
    )

    for j in real_nodes:
        model.addConstr(quicksum(x[i, j] for i in all_nodes if i != fictitious_node and i != j) == g[j])

    for j in real_nodes:
        model.addConstr(
            quicksum(x[i, j] for i in all_nodes if i != j) == quicksum(x[j, k] for k in all_nodes if k != j)
        )

    model.addConstr(quicksum(x[0, j] for j in real_nodes) == quicksum(x[i, fictitious_node] for i in real_nodes))

    Q_val = vehicle_capacity / percent_to_kg
    for i in all_nodes:
        for j in all_nodes:
            if i != j:
                model.addConstr(f[i, j] <= Q_val * x[i, j])

    for j in real_nodes:
        model.addConstr(
            quicksum(f[i, j] for i in all_nodes if i != j) + capped_fills[j] * g[j]
            == quicksum(f[j, k] for k in all_nodes if k != j)
        )

    for j in real_nodes:
        model.addConstr(f[0, j] == 0)

    N_count = len(all_nodes)
    for i in real_nodes:
        for j in real_nodes:
            if i != j:
                model.addConstr(u[i] - u[j] + N_count * x[i, j] <= N_count - 1)

    for i in real_nodes:
        model.addConstr(u[i] >= 1)
        model.addConstr(u[i] <= N_count)

    model.addConstr(quicksum(x[0, j] for j in real_nodes) <= number_vehicles)

    model.optimize()

    if model.Status == GRB.OPTIMAL or model.Status == GRB.TIME_LIMIT:
        vals = model.getAttr("x", x)
        succ = {}
        for i in all_nodes:
            for j in all_nodes:
                if i != j and vals[i, j] > 0.5:
                    succ[i] = j

        routes = []
        starts = [j for j in real_nodes if vals[0, j] > 0.5]
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
    """
    Look-ahead policy using Simulated Annealing (SANS).

    Args:
      data: Bin data (DataFrame or matrix).
      bins_coordinates: Dictionary of bin coordinates.
      distance_matrix: NxN distance matrix.
      params: SA parameters (T_init, iterations, alpha, T_min).
      must_go_bins: Bins identified for collection.
      values: Problem parameters.

    Returns:
      Tuple[List[List[int]], float, float]: Optimized routes, profit, and distance.
    """
    T_init, iterations_per_T, alpha, T_min, *_ = params

    density, V, vehicle_capacity = values["B"], values["E"], values["vehicle_capacity"]
    R, C, _ = values["R"], values["C"], values["Omega"]
    _, _, time_limit = 1, 1, values["time_limit"]

    iframe = isinstance(data, pd.DataFrame)
    if iframe:
        pass
    else:
        data = create_dataframe_from_matrix(data)
        data["#bin"] = data["#bin"] + 1

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

    all_bins_set = set(data["#bin"].tolist()) - {0}
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

    (
        optimized_routes,
        best_profit,
        last_distance,
        total_kg,
        last_revenue,
    ) = improved_simulated_annealing(
        single_route_solution,
        distance_matrix,
        time_limit,
        id_to_index,
        data,
        vehicle_capacity,
        T_init,
        T_min,
        alpha,
        iterations_per_T,
        R,
        V,
        density,
        C,
        must_go_bins,
        removed_bins=set(removed_bins),
        perc_bins_can_overflow=values.get("perc_bins_can_overflow", 0.0),
        volume=V,
        density_val=density,
        max_vehicles=1,
    )

    return optimized_routes, best_profit, last_distance


def policy_lookahead_hgs(current_fill_levels, binsids, must_go_bins, distance_matrix, values, coords):
    """
    Look-ahead policy using Hybrid Genetic Search (HGS).

    Identifies candidate bins and uses HGS to find an optimal collection route.

    Args:
      current_fill_levels: Current fill levels.
      binsids: List of bin IDs.
      must_go_bins: Bins that must be collected.
      distance_matrix: NxN distance matrix.
      values: Problem parameters.
      coords: Bin coordinates DataFrame.

    Returns:
      Tuple[List[int], float, float]: Tour, fitness, and cost.
    """
    B, E, Q = values["B"], values["E"], values["vehicle_capacity"]
    R, C = values["R"], values["C"]

    candidate_indices = [i for i, b_id in enumerate(binsids) if current_fill_levels[i] > 0 or b_id in must_go_bins]

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

    seed_df = pd.DataFrame({"#bin": bin_col, "Stock": stock_col, "Accum_Rate": accum_col})

    coord_dict_seed = {}
    for idx in bin_col:
        if idx < len(coords):
            lat = coords.iloc[idx]["Lat"]
            lng = coords.iloc[idx]["Lng"]
            coord_dict_seed[idx] = (lat, lng)
        else:
            coord_dict_seed[idx] = (0, 0)

    id_to_index_seed = {idx: idx for idx in bin_col}

    vrpp_routes = compute_initial_solution(seed_df, coord_dict_seed, distance_matrix, Q, id_to_index_seed)

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
        vrpp_tour_global,
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


def policy_lookahead_alns(
    current_fill_levels,
    binsids,
    must_go_bins,
    distance_matrix,
    values,
    coords,
    variant="custom",
):
    """
    Look-ahead policy using Adaptive Large Neighborhood Search (ALNS).

    Dispatches to one of the ALNS implementation variants (custom, package, or ortools).

    Args:
      current_fill_levels: Current fill levels.
      binsids: List of bin IDs.
      must_go_bins: Bins that must be collected.
      distance_matrix: NxN distance matrix.
      values: Problem parameters.
      coords: Bin coordinates DataFrame.
      variant: ALNS variant to use ('custom', 'package', 'ortools').

    Returns:
      Tuple[List[int], float, float]: Tour, dummy fitness (0), and cost.
    """
    B, E, Q = values["B"], values["E"], values["vehicle_capacity"]
    R, C = values["R"], values["C"]

    candidate_indices = [i for i, b_id in enumerate(binsids) if current_fill_levels[i] > 0 or b_id in must_go_bins]

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

    if variant == "package":
        routes, profit, cost = run_alns_package(distance_matrix, demands, Q, R, C, values)
    elif variant == "ortools":
        routes, profit, cost = run_alns_ortools(distance_matrix, demands, Q, R, C, values)
    else:
        routes, profit, cost = run_alns(distance_matrix, demands, Q, R, C, values)

    final_sequence = [0]
    for route in routes:
        final_sequence.extend(route)
        final_sequence.append(0)

    return final_sequence, profit, cost


def policy_lookahead_bcp(
    current_fill_levels,
    binsids,
    must_go_bins,
    distance_matrix,
    values,
    coords,
    env=None,
):
    """
    Look-ahead policy using Branch-Cut-and-Price (BCP).

    Args:
      current_fill_levels: Current fill levels.
      binsids: List of bin IDs.
      must_go_bins: Bins that must be collected.
      distance_matrix: NxN distance matrix.
      values: Problem parameters.
      coords: Bin coordinates DataFrame.
      env: Gurobi environment (if applicable).

    Returns:
      Tuple[List[int], float, float]: Tour, dummy fitness (0), and cost.
    """
    B, E, Q = values["B"], values["E"], values["vehicle_capacity"]
    R, C = values["R"], values["C"]

    candidate_indices = [i for i, b_id in enumerate(binsids) if current_fill_levels[i] > 0 or b_id in must_go_bins]

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

    routes, cost = run_bcp(
        distance_matrix,
        demands,
        Q,
        R,
        C,
        values,
        must_go_indices=global_must_go,
        env=None,
    )

    final_sequence = [0]
    for route in routes:
        final_sequence.extend(route)
        final_sequence.append(0)

    return final_sequence, 0, cost


def policy_lookahead_lk(
    current_fill_levels,
    binsids,
    must_go_bins,
    distance_matrix,
    values,
    coords,
):
    """
    Look-ahead policy using Lin-Kernighan heuristic.

    Args:
      current_fill_levels: Current fill levels.
      binsids: List of bin IDs.
      must_go_bins: Bins that must be collected.
      distance_matrix: NxN distance matrix.
      values: Problem parameters.
      coords: Bin coordinates DataFrame.

    Returns:
      Tuple[List[int], float, float]: Tour, dummy fitness (0), and cost.
    """
    B, E, Q = values["B"], values["E"], values["vehicle_capacity"]

    # Identify bins to collect: must_go + any positive fill (greedy candidates)
    # Similar to other policies, we collect must_go and potentially others.
    # For simplicity in this variant, we route the 'must_go_bins' and any non-empty bins.

    candidate_indices = [i for i, b_id in enumerate(binsids) if current_fill_levels[i] > 0 or b_id in must_go_bins]

    if not candidate_indices:
        return [0, 0], 0, 0

    # Build local distance matrix for candidates + depot (0)
    # Map local indices (0..k) to global bin IDs
    # Depot is always global 0.

    nodes_to_visit = [0] + [
        binsids[i] + 1 for i in candidate_indices
    ]  # binsids are 0-based, global IDs are 1-based (usually)
    # Actually, let's verify bin ID mapping.
    # consistently used: binsids[i] is the ID. Global usually is ID+1.
    # Let's map purely based on distance matrix indices.

    # Extract sub-matrix
    # distance_matrix is N+1 x N+1? (Depot + N bins)
    # binsids are 0..N-1.
    # Matrix indices: 0 is depot. i+1 is bin i.

    map_local_to_global = {0: 0}
    for idx, i in enumerate(candidate_indices):
        map_local_to_global[idx + 1] = i + 1

    n_nodes = len(nodes_to_visit)
    sub_matrix = np.zeros((n_nodes, n_nodes))

    for r in range(n_nodes):
        for c in range(n_nodes):
            global_r = map_local_to_global[r]
            global_c = map_local_to_global[c]
            sub_matrix[r, c] = distance_matrix[global_r][global_c]

    # Map global waste to local indices for solve_lk penalty calculation
    local_waste = np.zeros(n_nodes)
    for i, idx in enumerate(candidate_indices):
        fill = current_fill_levels[idx]
        local_waste[i + 1] = (fill / 100.0) * B * E

    # Solve TSP/VRP with LK (LKH-3 version)
    lk_tour_local, cost = solve_lk(sub_matrix, waste=local_waste, capacity=Q)

    # Map back to global IDs
    lk_tour_global = [map_local_to_global[i] for i in lk_tour_local]

    # If using LK for VRP, we must handle capacity.
    from .single_vehicle import get_multi_tour

    # Prepare global waste array for get_multi_tour
    max_id = max(max(map_local_to_global.values()), len(distance_matrix) - 1)
    waste_array = np.zeros(max_id + 1)
    for i in candidate_indices:
        fill = current_fill_levels[i]
        global_id = i + 1
        waste_array[global_id] = (fill / 100.0) * B * E

    # Check if capacity enforced
    if values.get("check_capacity", True):
        final_tour = get_multi_tour(lk_tour_global, waste_array, Q, distance_matrix)

    # Recalculate cost of final tour with splits
    final_cost = 0
    from .single_vehicle import get_route_cost

    final_cost = get_route_cost(distance_matrix, final_tour)

    return final_tour, 0, final_cost


@PolicyRegistry.register("policy_look_ahead")
class LookAheadPolicy(IPolicy):
    """
    Look-ahead policy class.
    Dispatches to various look-ahead strategies (VRPP, SANS, HGS, ALNS, BCP).
    """

    def execute(self, **kwargs: Any) -> Tuple[List[int], float, Any]:
        """
        Execute the look-ahead policy.
        """
        policy = kwargs["policy"]
        graph_size = kwargs["graph_size"]
        bins = kwargs["bins"]
        new_data = kwargs["new_data"]
        coords = kwargs["coords"]
        current_collection_day = kwargs["current_collection_day"]
        area = kwargs["area"]
        waste_type = kwargs["waste_type"]
        n_vehicles = kwargs["n_vehicles"]
        model_env = kwargs["model_env"]
        distance_matrix = kwargs["distance_matrix"]
        distancesC = kwargs["distancesC"]
        run_tsp = kwargs["run_tsp"]
        two_opt_max_iter = kwargs.get("two_opt_max_iter", 0)

        last_minute_config = kwargs.get("config", {}).get("lookahead", {})

        look_ahead_config = policy[policy.find("ahead_") + len("ahead_")]
        possible_configurations = {
            "a": [500, 75, 0.95, 0, 0.095, 0, 0],
            "b": [2000, 75, 0.7, 0, 0.095, 0, 0],
        }

        # Override from config if present
        if look_ahead_config in last_minute_config:
            possible_configurations[look_ahead_config] = last_minute_config[look_ahead_config]

        try:
            chosen_combination = possible_configurations[look_ahead_config]
        except KeyError:
            print("Possible policy_look_ahead configurations:")
            for pos_pol, configs in possible_configurations.items():
                print(f"{pos_pol} configuration: {configs}")
            raise ValueError(f"Invalid policy_look_ahead configuration: {policy}")

        binsids = np.arange(0, graph_size).tolist()
        must_go_bins = policy_lookahead(binsids, bins.c, bins.means, current_collection_day)

        tour = []
        cost = 0
        if len(must_go_bins) > 0:
            vehicle_capacity, R, B, C, E = load_area_and_waste_type_params(area, waste_type)
            values = {
                "R": R,
                "C": C,
                "E": E,
                "B": B,
                "vehicle_capacity": vehicle_capacity,
                "Omega": last_minute_config.get("Omega", 0.1),
                "delta": last_minute_config.get("delta", 0),
                "psi": last_minute_config.get("psi", 1),
            }
            routes = None
            if "vrpp" in policy:
                vrpp_la_config = last_minute_config.get("vrpp", {})
                routes, _, _ = policy_lookahead_vrpp(
                    bins.c,
                    binsids,
                    must_go_bins,
                    distance_matrix,
                    values,
                    number_vehicles=n_vehicles,
                    env=model_env,
                    time_limit=vrpp_la_config.get("time_limit", 60),
                )
            elif "sans" in policy:
                sans_config = last_minute_config.get("sans", {})
                values["time_limit"] = sans_config.get("time_limit", 60)
                values["perc_bins_can_overflow"] = sans_config.get("perc_bins_can_overflow", 0)

                T_min = sans_config.get("T_min", 0.01)
                T_init = sans_config.get("T_init", 75)
                iterations_per_T = sans_config.get("iterations_per_T", 5000)
                alpha = sans_config.get("alpha", 0.95)

                params = (T_init, iterations_per_T, alpha, T_min)
                new_data.loc[1:, "Stock"] = bins.c.astype("float32")
                new_data.loc[1:, "Accum_Rate"] = bins.means.astype("float32")
                routes, _, _ = policy_lookahead_sans(new_data, coords, distance_matrix, params, must_go_bins, values)
                if routes:
                    routes = routes[0]
            elif "hgs" in policy:
                hgs_config = last_minute_config.get("hgs", {})
                values["time_limit"] = hgs_config.get("time_limit", 60)
                routes, _, _ = policy_lookahead_hgs(bins.c, binsids, must_go_bins, distance_matrix, values, coords)
            elif "alns" in policy:
                alns_config = last_minute_config.get("alns", {})
                values["time_limit"] = alns_config.get("time_limit", 60)
                values["Iterations"] = alns_config.get("Iterations", 5000)
                variant = alns_config.get("variant", "default")

                if "package" in policy:
                    variant = "package"
                elif "ortools" in policy:
                    variant = "ortools"

                if alns_config.get("variant"):
                    variant = alns_config.get("variant")

                routes, _, _ = policy_lookahead_alns(
                    bins.c,
                    binsids,
                    must_go_bins,
                    distance_matrix,
                    values,
                    coords,
                    variant=variant,
                )
            elif "bcp" in policy:
                bcp_config = last_minute_config.get("bcp", {})
                values["time_limit"] = bcp_config.get("time_limit", 60)
                values["Iterations"] = bcp_config.get("Iterations", 50)
                routes, _, _ = policy_lookahead_bcp(bins.c, binsids, must_go_bins, distance_matrix, values, coords)
            elif "lkh" in policy:
                routes, _, _ = policy_lookahead_lk(bins.c, binsids, must_go_bins, distance_matrix, values, coords)
            else:
                values["shift_duration"] = 390
                values["perc_bins_can_overflow"] = 0
                points = create_points(new_data, coords)
                new_data.loc[1 : graph_size + 1, "Stock"] = (bins.c / 100).astype("float32")
                new_data.loc[1 : graph_size + 1, "Accum_Rate"] = (bins.means / 100).astype("float32")
                try:
                    routes, _, _ = find_solutions(
                        new_data,
                        coords,
                        distance_matrix,
                        chosen_combination,
                        must_go_bins,
                        values,
                        graph_size,
                        points,
                        time_limit=600,
                    )
                except Exception:
                    routes, _, _ = find_solutions(
                        new_data,
                        coords,
                        distance_matrix,
                        chosen_combination,
                        must_go_bins,
                        values,
                        graph_size,
                        points,
                        time_limit=3600,
                    )
                if routes:
                    routes = routes[0]

            if routes:
                tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
                if two_opt_max_iter > 0:
                    tour = local_search_2opt(tour, distance_matrix, two_opt_max_iter)
                cost = get_route_cost(distance_matrix, tour)
        else:
            tour = [0, 0]
            cost = 0
        return tour, cost, None
