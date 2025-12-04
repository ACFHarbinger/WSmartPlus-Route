import torch
import numpy as np

from src.utils.definitions import DAY_METRICS
from src.utils.log_utils import send_daily_output_to_gui
from src.utils.functions import move_to
from src.or_policies import (
    get_route_cost, find_route,
    create_points, find_solutions,
    policy_last_minute, policy_last_minute_and_path,
    policy_regular, policy_gurobi_vrpp, policy_hexaly_vrpp,
    policy_lookahead, policy_lookahead_vrpp, policy_lookahead_sans,
)
from .loader import load_area_and_waste_type_params


def set_daily_waste(model_data, waste, device, fill=None):
    model_data['waste'] = torch.as_tensor(waste, dtype=torch.float32).unsqueeze(0)/100.
    if 'fill_history' in model_data: 
        model_data['current_fill'] = torch.as_tensor(fill, dtype=torch.float32).unsqueeze(0)/100.
    return move_to(model_data, device)


def get_daily_results(total_collected, ncol, cost, tour, day, new_overflows, sum_lost, coordinates, profit):
    dlog = {key: 0 for key in DAY_METRICS}
    dlog['day'] = day
    dlog['overflows'] = new_overflows
    dlog['kg_lost'] = sum_lost
    if len(tour) > 2:
        rl_cost = new_overflows - total_collected + cost
        dlog['kg'] = total_collected
        dlog['ncol'] = ncol
        dlog['km'] = cost
        dlog['kg/km'] = total_collected / cost
        dlog['cost'] = rl_cost
        dlog['profit'] = profit
        ids = np.array([x for x in tour if x != 0])
        dlog['tour'] = [0] + coordinates.loc[ids, 'ID'].tolist() + [0]
    else:
        dlog['kg'] = 0
        dlog['ncol'] = 0
        dlog['km'] = 0
        dlog['kg/km'] = 0
        dlog['cost'] = new_overflows
        dlog['profit'] = 0
        dlog['tour'] = [0]
    return dlog


def run_day(graph_size, pol, bins, new_data, coords, run_tsp, sample_id,
            overflows, day, model_env, model_ls, n_vehicles, area, realtime_log_path,
            waste_type, distpath_tup, current_collection_day, cached, device, lock=None):
    cost = 0
    tour = []
    output_dict = None
    if bins.is_stochastic():
        new_overflows, fill, total_fill, sum_lost = bins.stochasticFilling()
    else:
        new_overflows, fill, total_fill, sum_lost = bins.loadFilling(day - 1)

    overflows += new_overflows
    distance_matrix, paths_between_states, dm_tensor, distancesC = distpath_tup
    policy = pol.rsplit('_', 1)[0]
    if 'policy_last_minute_and_path' in policy:
        cf = int(policy.rsplit("_and_path", 1)[1])
        if cf not in [50, 70, 90]:
            print('Valid cf values for policy_last_minute_and_path: [50, 70, 90]')
            raise ValueError(f'Invalid cf value for policy_last_minute_and_path: {cf}')
        bins.setCollectionLvlandFreq(cf=cf/100)
        tour = policy_last_minute_and_path(bins.c, distancesC, paths_between_states, bins.collectlevl, waste_type, area)
        cost = get_route_cost(distance_matrix, tour) if tour else 0
    elif 'policy_last_minute' in policy:
        cf = int(policy.rsplit("_last_minute", 1)[1])
        if cf not in [50, 70, 90]:
            print('Valid cf values for policy_last_minute: [50, 70, 90]')
            raise ValueError(f'Invalid cf value for policy_last_minute: {cf}')
        bins.setCollectionLvlandFreq(cf=cf/100)
        tour = policy_last_minute(bins.c, distancesC, bins.collectlevl, waste_type, area)
        cost = get_route_cost(distance_matrix, tour) if tour else 0
    elif 'policy_regular' in policy:
        lvl = int(policy.rsplit("_regular", 1)[1]) - 1
        if lvl not in [1, 2, 5]:
            print('Valid lvl values for policy_regular: [2, 3, 6]')
            raise ValueError(f'Invalid lvl value for policy_regular: {lvl + 1}')
        tour = policy_regular(bins.n, bins.c, distancesC, lvl, day, cached, waste_type, area)
        cost = get_route_cost(distance_matrix, tour) if tour else 0
        if cached is not None and not cached and tour: cached = tour
    elif policy[:2] == 'am' or policy[:4] == 'ddam' or "transgcn" in policy:
        model_data, graph, profit_vars = model_ls
        daily_data = set_daily_waste(model_data, bins.c, device, fill)
        tour, cost, output_dict = model_env.compute_simulator_day(daily_data, graph, dm_tensor, profit_vars, run_tsp)
    elif 'gurobi' in policy:
        gp_param = float(policy.rsplit("_vrpp", 1)[1])
        try:
            routes = policy_gurobi_vrpp(bins.c, distance_matrix.tolist(), model_env, gp_param, 
                                        bins.means, bins.std, waste_type, area, n_vehicles, time_limit=600)
        except:
            routes = policy_gurobi_vrpp(bins.c, distance_matrix.tolist(), model_env, gp_param, 
                                        bins.means, bins.std, waste_type, area, n_vehicles, time_limit=3600)

        if routes:
            tour = find_route(distancesC, np.array(routes[0])) if run_tsp else routes[0]
            cost = get_route_cost(distance_matrix, tour)
    elif 'hexaly' in policy:
        hex_param = float(policy.rsplit("_vrpp", 1)[1])
        try:
            routes = policy_hexaly_vrpp(bins.c, distance_matrix.tolist(), hex_param, bins.means, 
                                        bins.std, waste_type, area, n_vehicles, time_limit=600)
        except:
            routes = policy_hexaly_vrpp(bins.c, distance_matrix.tolist(), hex_param, bins.means, 
                                        bins.std, waste_type, area, n_vehicles, time_limit=3600)
        
        if routes:
            tour = find_route(distancesC, np.array(routes[0])) if run_tsp else routes[0]
            cost = get_route_cost(distance_matrix, tour)
    elif 'policy_look_ahead' in policy:
        look_ahead_config = policy[policy.find('ahead_') + len('ahead_')]
        possible_configurations = {
            'a': [500,75,0.95,0,0.095,0,0], 
            'b': [2000,75,0.7,0,0.095,0,0]
        }
        try:
            chosen_combination = possible_configurations[look_ahead_config]
        except KeyError as ke:
            print('Possible policy_look_ahead configurations:')
            for pos_pol, configs in possible_configurations.items():
                print(f'{pos_pol} configuration: {configs}')
            raise ValueError(f'Invalid policy_look_ahead configuration: {pol}')

        binsids = np.arange(0, graph_size).tolist()
        must_go_bins = policy_lookahead(binsids, bins.c, bins.means, current_collection_day)
        if len(must_go_bins) > 0:
            vehicle_capacity, R, B, C, E = load_area_and_waste_type_params(area, waste_type)
            values = {
                'R': R, 'C': C, 'E': E, 'B': B, 
                'vehicle_capacity': vehicle_capacity,
            }
            if 'vrpp' in policy:
                values['time_limit'] = 600
                fh = bins.get_fill_history().transpose()
                routes, profit, _ = policy_lookahead_vrpp(fh, binsids, must_go_bins, distance_matrix, values, env=model_env)
                if routes:
                    tour = find_route(distancesC, np.array(routes)) if run_tsp else routes
                    cost = get_route_cost(distance_matrix, tour)
            elif 'sans' in policy:
                values['time_limit'] = 60
                fh = bins.get_fill_history().transpose()
                T_min = 0.01
                T_init = 75
                iterations_per_T = 50000
                alpha = 0.7
                params = (T_init, iterations_per_T, alpha, T_min)
                routes, profit, _ = policy_lookahead_sans(fh, coords, distance_matrix, params, must_go_bins, values, binsids)
                if routes:
                    tour = find_route(distancesC, np.array(routes[0])) if run_tsp else routes[0]
                    cost = get_route_cost(distance_matrix, tour)
            else:
                values['shift_duration'] = 390 # minutes
                values['perc_bins_can_overflow'] = 0 # 0%
                points = create_points(new_data, coords)
                new_data.loc[1:graph_size+1, 'Stock'] = (bins.c/100).astype('float32')
                new_data.loc[1:graph_size+1, 'Accum_Rate'] = (bins.means/100).astype('float32')
                try:
                    routes, profit, removed_bins = find_solutions(new_data, coords, distance_matrix, chosen_combination,
                                                                must_go_bins, values, graph_size, points, time_limit=600)
                except:
                    routes, profit, removed_bins = find_solutions(new_data, coords, distance_matrix, chosen_combination,
                                                                must_go_bins, values, graph_size, points, time_limit=3600)
                
                if routes:
                    tour = find_route(distancesC, np.array(routes[0])) if run_tsp else routes[0]
                    cost = get_route_cost(distance_matrix, tour)
        else:
            tour = [0, 0]
            cost = 0
    else:
        raise ValueError("Unknown policy:", policy)
    
    collected, total_collected, ncol, profit = bins.collect(tour, cost)
    daily_log = get_daily_results(total_collected, ncol, cost, tour, day, new_overflows, sum_lost, coords, profit)
    send_daily_output_to_gui(daily_log, policy, sample_id, day, total_fill, collected, bins.c, realtime_log_path, tour, coords, lock)
    return (new_data, coords, bins), (overflows, daily_log, output_dict), cached
