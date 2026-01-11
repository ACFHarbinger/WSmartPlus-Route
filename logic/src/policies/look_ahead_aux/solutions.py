"""
Core solution generation and optimization framework for look-ahead policies.

Integrates construction heuristics, simulated annealing, and local search 
improvement loops to find high-quality routing configurations. Manages the 
lifecycle of solution discovery, from initial greedy builds to refined, 
multi-neighborhood optimized plans.
"""

import time
import math
import copy
import random
import numpy as np

from copy import deepcopy
from .sans_opt import (
    relocate_within_route, cross_exchange,
    or_opt_move, move_between_routes, mutate_route_by_swapping_bins,
    get_2opt_neighbors, insert_bin_in_route
)
from .move import (
    move_n_route_random, move_n_route_consecutive
)
from .swap import (
    swap_n_route_random, swap_n_route_consecutive
)
from .select import (
    insert_bins, remove_bins_end, remove_n_bins_random,
    remove_n_bins_consecutive, add_n_bins_random,
    add_n_bins_consecutive, add_route_with_removed_bins_random,
    add_route_with_removed_bins_consecutive
)
from .routes import uncross_arcs_in_routes, uncross_arcs_in_sans_routes
from .search import local_search, local_search_2, local_search_reversed
from .computations import compute_profit, compute_real_profit, compute_total_cost
from .check import (
    check_solution_admissibility, check_bins_overflowing_feasibility
)


# Lookahead base policy
def find_initial_solution(data, bins_coordinates, distance_matrix, number_of_bins, vehicle_capacity, E, B):
    """
    Construct a feasible initial solution for the routing problem.

    Processes overflow predictions and builds routes using a construction heuristic.

    Args:
        data (pd.DataFrame): Bin weights and metadata.
        bins_coordinates (List): Lat/Lon pairs.
        distance_matrix (np.ndarray): Shortest path distances.
        number_of_bins (int): Scale of the problem.
        vehicle_capacity (float): Max tanker capacity.
        E (float): Bin volume.
        B (float): Bin density.

    Returns:
        Tuple: (routes, removed_bins, bins_cannot_removed, points).
    """
    bins = list(data['#bin'][1:number_of_bins+1])
    depot = list(data['#bin'][0:1])

    # Initialization of routes
    i = 0
    routes_list = list()
    max_lng = max(bins_coordinates['Lng'])
    min_lng = min(bins_coordinates['Lng'])
    lng_amp = max_lng - min_lng
    zone_1_s = min_lng
    zone_1_l = min_lng + lng_amp / 3
    zone_2_l = min_lng + 2 * lng_amp / 3
    lng_list = list(bins_coordinates['Lng'])
    lng_list_without_dep = lng_list[1:len(lng_list)]
    bins_zone_1 = []
    bins_zone_2 = []
    bins_zone_3 = []
    for n, h in enumerate(lng_list_without_dep):
        if h >= zone_1_s and h < zone_1_l:
            bins_zone_1.append(n+1)
        elif h >= zone_1_l and h < zone_2_l:
            bins_zone_2.append(n+1)
        else:
            bins_zone_3.append(n+1)
    while len(bins) != 0:
        i += 1
        globals()['route_{0}'.format(i)] = []
        space_occupied = 0

        # Choose depot to initialize the route
        bin_chosen_n = depot[0]

        # Get data for the bin chosen
        corresponding_row = data[data['#bin'] == bin_chosen_n]
        
        # Get fill level of the bin (stock)
        stock = (corresponding_row.iloc[0]['Stock'] + corresponding_row.iloc[0]['Accum_Rate']) * E * B
        space_occupied += stock
        globals()['route_{0}'.format(i)].append(bin_chosen_n)

        # Previous_bin is the bin previously added in the route
        previous_bin = bin_chosen_n

        # While there is space in the vehicle and there are bins to collect:
        while space_occupied < vehicle_capacity and len(bins) != 0:
            while space_occupied < vehicle_capacity and len(bins_zone_1) != 0:
                # Get the distance between the previous bin and all the other bins
                row = distance_matrix[previous_bin][:]
                
                # Delete previous bin to previous bin distance
                row_new = np.delete(row,previous_bin)
                
                # Get the index of the bin with the minimum distance from the previous one
                min(row_new)
                min_idx = row_new.argmin()

                # Get the correct bin id (because of deleting the 0 distance)
                if min_idx >= previous_bin:
                    next_bin_idx = min_idx + 1
                else:
                    next_bin_idx = min_idx

                # Check if the closest bin is already in any of the routes created
                if next_bin_idx in globals()['route_{0}'.format(i)] or next_bin_idx not in bins or next_bin_idx not in bins_zone_1:
                    # if yes, sort the distances from the shortest to the longest
                    row_sorted = np.sort(row_new)
                    j = 1
                    stop = 'A'

                    # Iterate through sorted list until the closest bin that is not in any route is found
                    while j <= len(row_sorted[1:]) and stop != 'B':
                        row_new_list = row_new.tolist()

                        # Get index of the bin that is being tried
                        idx_tried_bin = row_new_list.index(row_sorted[j])
                        if idx_tried_bin >= previous_bin:
                            next_bin_idx = idx_tried_bin + 1
                        else:
                            next_bin_idx = idx_tried_bin
                        
                        # Verify if it is in any route and not only in the current route
                        if next_bin_idx not in globals()['route_{0}'.format(i)] and next_bin_idx in bins and next_bin_idx in bins_zone_1:
                            stop = 'B'
                            min_idx = next_bin_idx

                        j += 1

                # Get current bin index from the bins list
                bin_index_in_bins = bins.index(next_bin_idx)
                
                # Update space occupied in the vehicle
                corresponding_row = data[data['#bin'] == next_bin_idx]
                stock = (corresponding_row.iloc[0]['Stock'] + corresponding_row.iloc[0]['Accum_Rate']) * E * B
                space_occupied += stock
                if space_occupied < vehicle_capacity and next_bin_idx not in globals()['route_{0}'.format(i)]:
                    # Add bin to the route
                    globals()['route_{0}'.format(i)].append(next_bin_idx)
                    bins.pop(bin_index_in_bins)

                    # Remove bin from bins zone
                    bin_index_in_bins_zone = bins_zone_1.index(next_bin_idx)
                    bins_zone_1.pop(bin_index_in_bins_zone)

                    # Update previous bin
                    previous_bin = next_bin_idx
            while space_occupied < vehicle_capacity and len(bins_zone_2) != 0:
                # Get the distance between the previous bin and all the other bins
                row = distance_matrix[previous_bin][:]

                # Delete previous bin to previous bin distance
                row_new = np.delete(row,previous_bin)
                
                # Get the index of the bin with the minimum distance from the previous one
                min(row_new)
                min_idx = row_new.argmin()

                # Get the correct bin id (because of deleting the 0 distance)
                if min_idx >= previous_bin:
                    next_bin_idx = min_idx + 1
                else:
                    next_bin_idx = min_idx

                # Check if the closest bin is already in any of the routes created
                if next_bin_idx in globals()['route_{0}'.format(i)] or next_bin_idx not in bins or next_bin_idx not in bins_zone_2:
                    # if yes, sort the distances from the shortest to the longest
                    row_sorted = np.sort(row_new)
                    j = 1
                    stop = 'A'

                    # Iterate through sorted list until the closest bin that is not in any route is found
                    while j <= len(row_sorted[1:]) and stop != 'B':
                        row_new_list = row_new.tolist()

                        # Get index of the bin that is being tried
                        idx_tried_bin = row_new_list.index(row_sorted[j])
                        if idx_tried_bin >= previous_bin:
                            next_bin_idx = idx_tried_bin + 1
                        else:
                            next_bin_idx = idx_tried_bin
                        
                        # Verify if it is in any route and not only in the current route
                        if next_bin_idx not in globals()['route_{0}'.format(i)] and next_bin_idx in bins and next_bin_idx in bins_zone_2:
                            stop = 'B'
                            min_idx = next_bin_idx

                        j += 1

                # Get current bin index from the bins list
                bin_index_in_bins = bins.index(next_bin_idx)
                
                # Update space occupied in the vehicle
                corresponding_row = data[data['#bin'] == next_bin_idx]
                stock = (corresponding_row.iloc[0]['Stock'] + corresponding_row.iloc[0]['Accum_Rate']) * E * B
                space_occupied += stock
                if space_occupied < vehicle_capacity and next_bin_idx not in globals()['route_{0}'.format(i)]:
                    # Add bin to the route
                    globals()['route_{0}'.format(i)].append(next_bin_idx)
                    
                    # Remove bin from bins to be collected list
                    bins.pop(bin_index_in_bins)
                    
                    # Remove bin from bins zone
                    bin_index_in_bins_zone = bins_zone_2.index(next_bin_idx)
                    bins_zone_2.pop(bin_index_in_bins_zone)
                    
                    # Update previous bin
                    previous_bin = next_bin_idx

            while space_occupied < vehicle_capacity and len(bins_zone_3) != 0:
                # Get the distance between the previous bin and all the other bins
                row = distance_matrix[previous_bin][:]
                row_sorted = np.sort(row)
                j = 1
                stop = 'A'

                # Iterate through sorted list until the closest bin that is not in any route is found
                while j <= len(row_sorted[1:]) and stop != 'B':
                    _ = row_sorted.tolist()
                    possible_indexes = [index for index, value in enumerate(row) if value == row_sorted[j]]
                    for z in possible_indexes:
                        if z in bins:
                            idx_tried_bin = z
                            next_bin_idx = z
                    
                    # Verify if it is in any route and not only in the current route
                    if next_bin_idx not in globals()['route_{0}'.format(i)] and next_bin_idx in bins and next_bin_idx in bins_zone_3:
                        stop = 'B'
                        min_idx = next_bin_idx

                    j += 1

                # Get current bin index from the bins list
                bin_index_in_bins = bins.index(next_bin_idx)
                
                # Update space occupied in the vehicle
                corresponding_row = data[data['#bin'] == next_bin_idx]
                stock = (corresponding_row.iloc[0]['Stock'] + corresponding_row.iloc[0]['Accum_Rate']) * E * B
                space_occupied += stock
                if space_occupied < vehicle_capacity and next_bin_idx not in globals()['route_{0}'.format(i)]:
                    # Add bin to the route
                    globals()['route_{0}'.format(i)].append(next_bin_idx)
                    
                    # Remove bin from bins to be collected list
                    bins.pop(bin_index_in_bins)
                    
                    # Remove bin from bins zone
                    bin_index_in_bins_zone = bins_zone_3.index(next_bin_idx)
                    bins_zone_3.pop(bin_index_in_bins_zone)
                    
                    # Update previous bin
                    previous_bin = next_bin_idx
                #else:
                    #travel_time = travel_time - (distance/37.5) * 60 + 5

        globals()['route_{0}'.format(i)] = globals()['route_{0}'.format(i)] + depot
        routes_list.append(globals()['route_{0}'.format(i)])
    return routes_list


def find_solutions(data, bins_coordinates, distance_matrix, chosen_combination, must_go_bins, values, n_bins, points, time_limit):
    """
    Find high-quality routing solutions using a randomized local search procedure.

    Args:
        data (pd.DataFrame): Bin and weight context.
        bins_coordinates (List): Locations.
        distance_matrix (np.ndarray): distances.
        chosen_combination (Tuple): Parameter set (vehicle penalty, load penalty, etc).
        must_go_bins (List[int]): Nodes that must be visited.
        values (Dict): Global constants.
        n_bins (int): Problem size.
        points (Dict): Coordinates map.
        time_limit (float): Max execution time.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    number_iterations = chosen_combination[0]
    T_initial = chosen_combination[1]
    T_param = chosen_combination[2]
    p_vehicle = chosen_combination[3]
    p_load = chosen_combination[4]
    p_route_difference = chosen_combination[5]
    p_shift = chosen_combination[6]
    initial_solution = find_initial_solution(data, bins_coordinates, distance_matrix, n_bins, values['vehicle_capacity'], values['E'], values['B'])
    initial_solution, _, _ = uncross_arcs_in_routes(initial_solution, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Simulated annealing
    removed_bins = []
    bins_to_remove_random = []
    bins_to_remove_consecutive = []
    bins_to_add_random = []
    bins_to_add_consecutive = []
    number_iterations = chosen_combination[0]
    T_initial = chosen_combination[1]
    T_param = chosen_combination[2]
    p_vehicle = chosen_combination[3]
    p_load = chosen_combination[4]
    p_route_difference = chosen_combination[5]
    p_shift = chosen_combination[6]

    initial_sol = initial_solution
    previous_sol = initial_sol
    initial_sol_profit = compute_profit(initial_sol, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
    _ = compute_real_profit(initial_sol, p_vehicle, data, distance_matrix, values)
    previous_sol_profit = initial_sol_profit
    routes_list = deepcopy(initial_solution)
    count = 0
    tic = time.perf_counter()
    for i in range (1,number_iterations + 1):
        chosen_procedure, _, bin_to_remove, bin_to_add, bins_to_remove_random, bins_to_remove_consecutive, bins_to_add_random, \
        bins_to_add_consecutive, bins_random, bins_consecutive = local_search(routes_list, removed_bins, distance_matrix, must_go_bins)
        count = count + 1

        # Compute profit for the new solution
        current_sol_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

        # Check candidate route feasibility
        status = check_bins_overflowing_feasibility(data,routes_list, n_bins, values['perc_bins_can_overflow'], values['E'], values['B'])

        # Check candidate route admissibility
        _ = check_solution_admissibility(routes_list, removed_bins, n_bins)
        if status == 'pass':
            # Compare current profit with previous profit
            delta = current_sol_profit - previous_sol_profit
            if delta > 0:
                previous_sol = deepcopy(routes_list)
                previous_sol_profit = current_sol_profit
            elif delta == 0:
                if chosen_procedure == 'Drop bin' and bin_to_remove is not None:
                    removed_bins.remove(bin_to_remove)
                if chosen_procedure == 'Add bin' and bin_to_add is not None:
                    removed_bins.append(bin_to_add)
                if chosen_procedure == 'Remove n bins random' and bins_to_remove_random != []:
                    for n in bins_to_remove_random:
                        removed_bins.remove(n)
                if chosen_procedure == 'Remove n bins consecutive' and bins_to_remove_consecutive != []:
                    for o in bins_to_remove_consecutive:
                        removed_bins.remove(o)
                if chosen_procedure == 'Add n bins random' and bins_to_add_random != []:
                    for r in bins_to_add_random:
                        removed_bins.append(r)
                if chosen_procedure == 'Add n bins consecutive' and bins_to_add_consecutive != []:
                    for s in bins_to_add_consecutive:
                        removed_bins.append(s)
                if chosen_procedure == 'Add route with removed bins random':
                    for t in bins_random:
                        removed_bins.append(t)
                if chosen_procedure == 'Add route with removed bins consecutive':
                    for u in bins_consecutive:
                        removed_bins.append(u)
            else:
                T = T_initial / (i**T_param)
                p = math.exp(delta/T)
                p_random = np.random.uniform(low=0.0, high=1.0, size=None)
                if p >= p_random:
                    previous_sol = deepcopy(routes_list)
                    previous_sol_profit = current_sol_profit
                else:
                    if chosen_procedure == 'Drop bin' and bin_to_remove is not None:
                        removed_bins.remove(bin_to_remove)
                    if chosen_procedure == 'Add bin' and bin_to_add is not None:
                        removed_bins.append(bin_to_add)
                    if chosen_procedure == 'Remove n bins random' and bins_to_remove_random != []:
                        for n in bins_to_remove_random:
                            removed_bins.remove(n)
                        bins_to_remove_random = []
                    if chosen_procedure == 'Remove n bins consecutive' and bins_to_remove_consecutive != []:
                        for o in bins_to_remove_consecutive:
                            removed_bins.remove(o)
                    if chosen_procedure == 'Add n bins random' and bins_to_add_random != []:
                        for r in bins_to_add_random:
                            removed_bins.append(r)
                    if chosen_procedure == 'Add n bins consecutive' and bins_to_add_consecutive != []:
                        for s in bins_to_add_consecutive:
                            removed_bins.append(s)
                    if chosen_procedure == 'Add route with removed bins random':
                        for t in bins_random:
                            removed_bins.append(t)
                    if chosen_procedure == 'Add route with removed bins consecutive':
                        for u in bins_consecutive:
                            removed_bins.append(u)
        if status == 'fail':
            if chosen_procedure == 'Drop bin' and bin_to_remove is not None:
                removed_bins.remove(bin_to_remove)
            if chosen_procedure == 'Add bin' and bin_to_add is not None:
                removed_bins.append(bin_to_add)
            if chosen_procedure == 'Remove n bins random' and bins_to_remove_random != []:
                for n in bins_to_remove_random:
                    removed_bins.remove(n)
            if chosen_procedure == 'Remove n bins consecutive' and bins_to_remove_consecutive != []:
                for o in bins_to_remove_consecutive:
                    removed_bins.remove(o)
            if chosen_procedure == 'Add n bins random' and bins_to_add_random != []:
                for r in bins_to_add_random:
                    removed_bins.append(r)
            if chosen_procedure == 'Add n bins consecutive' and bins_to_add_consecutive != []:
                for s in bins_to_add_consecutive:
                    removed_bins.append(s)
            if chosen_procedure == 'Add route with removed bins random':
                for t in bins_random:
                    removed_bins.append(t)
            if chosen_procedure == 'Add route with removed bins consecutive':
                for u in bins_consecutive:
                    removed_bins.append(u)

        routes_list = deepcopy(previous_sol)
        _ = compute_real_profit(previous_sol, p_vehicle,data, distance_matrix, values)
        if (time.perf_counter()-tic) > time_limit:
            break

    SA_sol = deepcopy(previous_sol)
    _ = compute_real_profit(SA_sol, p_vehicle, data, distance_matrix, values)
    # end SA

    # Uncross after SA
    solution_after_uncross_1, _, _ = uncross_arcs_in_routes(SA_sol, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # First local search
    solution_after_LS_1, _, _ = local_search_2(solution_after_uncross_1, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after first local search
    solution_after_uncross_2, _, _ = uncross_arcs_in_routes(solution_after_LS_1, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # First reversed local search
    solution_after_reversed_LS_1, _, _ = local_search_reversed(solution_after_uncross_2, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after first reversed local search
    solution_after_uncross_3, _, _ = uncross_arcs_in_routes(solution_after_reversed_LS_1, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Second local search
    solution_after_LS_2, _, _ = local_search_2(solution_after_uncross_3, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after second local search
    solution_after_uncross_4, _, _ = uncross_arcs_in_routes(solution_after_LS_2, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Second reversed local search
    solution_after_reversed_LS_2, _, _ = local_search_reversed(solution_after_uncross_4, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after second reversed local search
    solution_after_uncross_5, _, _ = uncross_arcs_in_routes(solution_after_reversed_LS_2, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Third local search
    solution_after_LS_3, _, _ = local_search_2(solution_after_uncross_5, p_vehicle,p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after third local search
    solution_after_uncross_6, _, _ = uncross_arcs_in_routes(solution_after_LS_3, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Third reversed local search
    solution_after_reversed_LS_3, _, _ = local_search_reversed(solution_after_uncross_6, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after third reversed local search
    solution_after_uncross_7, _, __ = uncross_arcs_in_routes(solution_after_reversed_LS_3, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Fourth local search
    solution_after_LS_4, _, _ = local_search_2(solution_after_uncross_7, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after fourth local search
    solution_after_uncross_8, _, _ = uncross_arcs_in_routes(solution_after_LS_4, p_vehicle,p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Fourth reversed local search
    solution_after_reversed_LS_4, _, _ = local_search_reversed(solution_after_uncross_8, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after fourth reversed local search
    solution_after_uncross_9, _, _ = uncross_arcs_in_routes(solution_after_reversed_LS_4,p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Fifth local search
    solution_after_LS_5, _, _ = local_search_2(solution_after_uncross_9, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after fifth local search
    solution_after_uncross_10, _, _ = uncross_arcs_in_routes(solution_after_LS_5, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Fifth reversed local search
    solution_after_reversed_LS_5, _, _ = local_search_reversed(solution_after_uncross_10, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Uncross after fifth reversed local search
    solution_after_uncross_11, _, _ = uncross_arcs_in_routes(solution_after_reversed_LS_5, p_vehicle, p_load, p_route_difference, p_shift, data, points, distance_matrix, values)

    # Remove bins local search - 1
    solution_after_remove, _, _, removed_bins = remove_bins_end(solution_after_uncross_11, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  must_go_bins, distance_matrix, values)

    # Insert bins local search - 1
    solution_after_insert, _, _, removed_bins = insert_bins(solution_after_remove, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 2
    solution_after_remove_2, _, _, removed_bins = remove_bins_end(solution_after_insert, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 2
    solution_after_insert_2, _, _, removed_bins = insert_bins(solution_after_remove_2, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 3
    solution_after_remove_3, _, _, removed_bins = remove_bins_end(solution_after_insert_2, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 3
    solution_after_insert_3, _, _, removed_bins = insert_bins(solution_after_remove_3, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 4
    solution_after_remove_4, _, _, removed_bins = remove_bins_end(solution_after_insert_3, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 4
    solution_after_insert_4, _, _, removed_bins = insert_bins(solution_after_remove_4, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 5
    solution_after_remove_5, _, _, removed_bins = remove_bins_end(solution_after_insert_4, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 5
    solution_after_insert_5, _, _, removed_bins = insert_bins(solution_after_remove_5, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local6
    solution_after_remove_6, _, _, removed_bins = remove_bins_end(solution_after_insert_5, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 6
    solution_after_insert_6, _, _, removed_bins = insert_bins(solution_after_remove_6, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 7
    solution_after_remove_7, _, _, removed_bins = remove_bins_end(solution_after_insert_6, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 7
    solution_after_insert_7, _, _, removed_bins = insert_bins(solution_after_remove_7, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)

    # Remove bins local search - 8
    solution_after_remove_8, _, _, removed_bins = remove_bins_end(solution_after_insert_7, removed_bins, p_vehicle,p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 8
    solution_after_insert_8, _, _, removed_bins = insert_bins(solution_after_remove_8, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 9
    solution_after_remove_9, _, _, removed_bins = remove_bins_end(solution_after_insert_8, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 9
    solution_after_insert_9, _, _, removed_bins = insert_bins(solution_after_remove_9, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    # Remove bins local search - 10
    solution_after_remove_10, _, _, removed_bins = remove_bins_end(solution_after_insert_9, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data, must_go_bins, distance_matrix, values)

    # Insert bins local search - 10
    solution_after_insert_10, _, real_profit_after_insert_10, removed_bins = insert_bins(solution_after_remove_10, removed_bins, p_vehicle, p_load, p_route_difference, p_shift, data,  distance_matrix, values)

    routes = solution_after_insert_10
    profit = real_profit_after_insert_10
    return routes, profit, removed_bins


# Lookahead sans policy
def compute_initial_solution(data, bins_coordinates, distance_matrix, vehicle_capacity, id_to_index):
    """
    Simplified initial solution builder for SANS policy.

    Args:
        data (pd.DataFrame): Bin metadata.
        bins_coordinates (List): Coordinates.
        distance_matrix (np.ndarray): Distances.
        vehicle_capacity (float): Tanker capacity.
        id_to_index (Dict): Mapping.

    Returns:
        Tuple: (initial_routes, removed_bins).
    """
    depot = data['#bin'].iloc[0]
    all_bins = list(data['#bin'][1:])
    bins_coordinates = {b: coord for b, coord in bins_coordinates.items() if b in all_bins or b == depot}
    stocks = dict(zip(data['#bin'], data['Stock']))

    rotas = []
    bins_restantes = set(all_bins)

    # Enquanto houver bins a serem coletados
    while bins_restantes:
        # Inicializa a rota a partir do depósito
        current_route = [depot]
        carga = 0
        atual = depot
        bins_restantes_inicial = set(bins_restantes)

        # Enquanto houver bins restantes e espaço no veículo
        while bins_restantes_inicial:
            # Filtra os bins que podem ser adicionados sem ultrapassar a capacidade do veículo
            candidatos = [
                b for b in bins_restantes_inicial
                if carga + stocks[b] <= vehicle_capacity
            ]
            if not candidatos:
                break

            # Encontra o bin mais próximo
            idx_atual = id_to_index[atual]
            proximo = min(
                candidatos,
                key=lambda b: distance_matrix[idx_atual][id_to_index[b]]
            )

            # Adiciona o bin mais próximo à rota
            current_route.append(proximo)
            carga += stocks[proximo]
            bins_restantes_inicial.remove(proximo)
            atual = proximo

        # Retorna ao depósito
        current_route.append(depot)

        # Se a rota tiver mais de 2 pontos (depot + bins), adiciona à lista de rotas
        if len(current_route) > 2:
            rotas.append(current_route)

        # Remove os bins que já foram coletados
        bins_restantes = bins_restantes - set(current_route[1:-1])
    return rotas


def improved_simulated_annealing(
        routes, distance_matrix, time_limit, id_to_index, data, vehicle_capacity,
        T_init=1000, T_min=0.001, alpha=0.995, iterations_per_T=100, R=0.165, V=2.5, 
        density=20, C=1.0, must_go_bins=None, removed_bins=None, verbose=False,
        perc_bins_can_overflow=0.0, volume=2.5, density_val=20, max_vehicles=None
    ):
    """
    Refine routing solutions using a multi-neighborhood Simulated Annealing algorithm.

    Args:
        routes (List[List[int]]): Initial routes.
        distance_matrix (np.ndarray): Distances.
        time_limit (float): Max runtime.
        id_to_index (Dict): Node ID mapping.
        data (pd.DataFrame): Dataset.
        vehicle_capacity (float): Vehicle limit.
        T_init (float): Start temperature.
        T_min (float): Stop temperature.
        alpha (float): Cooling rate.
        iterations_per_T (int): Steps at each temperature level.
        R (float): Revenue factor.
        V (float): Bin volume.
        density (float): Waste density.
        C (float): Distance cost factor.
        must_go_bins (List): Invariant nodes.
        removed_bins (List): Uncollected nodes.
        verbose (bool): Log progress.
        perc_bins_can_overflow (float): Threshold.
        volume (float): Bin size.
        density_val (float): Density.
        max_vehicles (int): Hard fleet limit.

    Returns:
        Tuple: (best_routes, best_removed_bins, stats_history).
    """
    def _power_function_decay(T_init, i, T_param):
        return T_init / (i ** T_param)
    
    start_time = time.time()
    
    # --- 1. ROBUST INITIALIZATION ---
    if removed_bins is None:
        removed_bins = set()
    else:
        removed_bins = set(removed_bins)
        
    if must_go_bins is None:
        must_go_bins = set()
    else:
        must_go_bins = set(must_go_bins)

    # Capture any bins missed by the initial greedy solution
    all_bins_ids = set(data['#bin'].tolist()) - {0}
    scheduled_bins = set()
    for r in routes:
        for b in r:
            if b != 0:
                scheduled_bins.add(b)
    missing_bins = all_bins_ids - scheduled_bins
    for b in missing_bins:
        removed_bins.add(b)

    initial_solution = copy.deepcopy(routes)
    current_solution = uncross_arcs_in_sans_routes(initial_solution, id_to_index, distance_matrix)
    
    # Calculate initial metrics
    current_cost = compute_total_cost(current_solution, distance_matrix, id_to_index)
    # Calculate initial metrics
    current_cost = compute_total_cost(current_solution, distance_matrix, id_to_index)
    
    # Calculate STRICT SIMULATOR PROFIT (Route 0 Only + Capacity Cutoff)
    # 1. Total KM Cost (We pay for everything planned)
    # compute_total_cost returns total km? No, it returns cost = km * C? 
    # check solutions.py imports... compute_total_cost.
    # Actually compute_total_profit returns total_km. 
    # Let's call compute_total_profit just to get total_km and naive revenue if we want, or just use current_cost.
    # current_cost IS the transportation cost.
    
    # 2. Real Revenue (Route 0 Cutoff)
    real_kg = 0
    collected_must_go = set()
    if len(current_solution) > 0 and len(current_solution[0]) > 2:
        route0 = current_solution[0]
        stocks = dict(zip(data['#bin'], data['Stock']))
        current_load = 0
        for b in route0:
            if b == 0:
                continue
            bin_kg = stocks.get(b, 0) * V * density / 100.0
            
            if current_load + bin_kg <= vehicle_capacity:
                current_load += bin_kg
                real_kg += bin_kg
                if must_go_bins and b in must_go_bins:
                    collected_must_go.add(b)
            else:
                break
    
    current_revenue = real_kg * R
    
    # Must-Go Penalty: Force all must_go_bins into the valid trunk of Route 0
    missed_must_go = len(must_go_bins) - len(collected_must_go) if must_go_bins else 0
    penalty_must_go = missed_must_go * 10000.0 # Huge penalty
    
    current_profit = current_revenue - current_cost - penalty_must_go

    initial_receita = current_revenue
    best_solution = copy.deepcopy(current_solution)
    best_profit = current_profit
    last_receita = initial_receita
    last_weight = real_kg
    last_distance = current_cost
    
    no_improvement_count = 0
    T = T_init
    stocks = dict(zip(data['#bin'], data['Stock']))

    iter_count = 0
    while T > T_min:
        if time.time() - start_time > time_limit:
            if verbose:
                print("[DEBUG] Time limit reached.")
            break

        for _ in range(iterations_per_T):
            iter_count += 1
            if time.time() - start_time > time_limit:
                break
            
            # --- 2. STATE COPYING (Fixes Missing Bin 100) ---
            new_solution = copy.deepcopy(current_solution)
            candidate_removed_bins = copy.deepcopy(removed_bins)
            
            # --- 3. GUARD CLAUSES (Fixes IndexError Crash) ---
            # Separate operators into those needing routes and those needing removed bins
            route_ops = [
                "2opt", "move", "swap", "remove", "insert",
                "move_n_random", "move_n_consec", "swap_n_random", "swap_n_consec",
                "remove_n_bins", "remove_n_bins_consec", "relocate", "cross", "or-opt"
            ]
            
            add_ops = [
                "add_n_bins", "add_n_bins_consec",
                "add_route_removed", "add_route_removed_consec"
            ]

            # Smart Selection: If no routes, MUST add. If no removed bins, MUST modify routes.
            valid_ops = []
            if len(new_solution) > 0:
                valid_ops.extend(route_ops)
            if len(candidate_removed_bins) > 0:
                valid_ops.extend(add_ops)
            
            if not valid_ops:
                # Dead end (no routes and no bins to add), break or continue
                continue

            op = random.choice(valid_ops)

            # --- 4. APPLY OPERATORS ---
            # (Note: We pass candidate_removed_bins to operators, protecting the main set)
            if op == "2opt":
                # Guard: Route must be long enough
                valid_indices = [i for i, r in enumerate(new_solution) if len(r) > 3]
                if valid_indices:
                    r = random.choice(valid_indices)
                    new_solution[r] = random.choice(get_2opt_neighbors(new_solution[r]))
            elif op == "move":
                neighbors = move_between_routes(new_solution, data, vehicle_capacity, id_to_index)
                if neighbors:
                    new_solution = random.choice(neighbors)
            elif op == "swap":
                r = random.choice(range(len(new_solution)))
                new_solution[r] = mutate_route_by_swapping_bins(new_solution[r], num_bins=random.choice([1, 2]))
            elif op == "remove":
                # Assuming remove_n_bins_random handles n=1 internally or we call specific func
                new_solution = remove_n_bins_random(new_solution, candidate_removed_bins, must_go_bins, n=1)
            elif op == "move_n_random":
                new_solution = move_n_route_random(new_solution, n=random.randint(2, 5))
            elif op == "move_n_consec":
                new_solution = move_n_route_consecutive(new_solution, n=random.randint(2, 5))
            elif op == "swap_n_random":
                new_solution = swap_n_route_random(new_solution, n=random.randint(2, 5))
            elif op == "swap_n_consec":
                new_solution = swap_n_route_consecutive(new_solution, n=random.randint(2, 5))
            elif op == "remove_n_bins":
                new_solution = remove_n_bins_random(new_solution, candidate_removed_bins, must_go_bins, n=random.randint(2, 5))
            elif op == "remove_n_bins_consec":
                new_solution = remove_n_bins_consecutive(new_solution, candidate_removed_bins, must_go_bins, n=random.randint(2, 5))
            elif op == "add_n_bins":
                new_solution = add_n_bins_random(new_solution, candidate_removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, n=2)
            elif op == "add_n_bins_consec":
                new_solution = add_n_bins_consecutive(new_solution, candidate_removed_bins, stocks, vehicle_capacity, id_to_index, distance_matrix, n=2)
            elif op == "add_route_removed":
                new_solution = add_route_with_removed_bins_random(new_solution, candidate_removed_bins, stocks, vehicle_capacity)
            elif op == "add_route_removed_consec":
                new_solution = add_route_with_removed_bins_consecutive(new_solution, candidate_removed_bins, stocks, vehicle_capacity)
            elif op == "relocate":
                r = random.choice(range(len(new_solution)))
                new_solution[r] = relocate_within_route(new_solution[r])
            elif op == "cross":
                new_solution = cross_exchange(new_solution)
            elif op == "or-opt":
                r = random.choice(range(len(new_solution)))
                new_solution[r] = or_opt_move(new_solution[r])
            elif op == "insert":
                if not new_solution:
                    continue
                r = random.choice(range(len(new_solution)))
                all_bins = set(data['#bin']) - {0}
                used_bins = set(b for route in new_solution for b in route)
                unused = list(all_bins - used_bins)
                if unused:
                    bin_to_insert = random.choice(unused)
                    load = sum(stocks.get(b, 0) for b in new_solution[r] if b != 0)
                    if load + stocks.get(bin_to_insert, 0) <= vehicle_capacity:
                        new_solution[r] = insert_bin_in_route(new_solution[r], bin_to_insert, id_to_index, distance_matrix)
                        # Remove from candidate removed bins if it was there
                        if bin_to_insert in candidate_removed_bins:
                            candidate_removed_bins.remove(bin_to_insert)

            # Filter out empty routes (safely)
            new_solution = [r for r in new_solution if len(r) > 2]
            
            # --- 5. EVALUATION ---
            # --- 5. EVALUATION ---
            new_cost = compute_total_cost(new_solution, distance_matrix, id_to_index)
            # Calculate STRICT SIMULATOR PROFIT (Route 0 Only + Capacity Cutoff)
            real_kg = 0
            collected_must_go = set()
            if len(new_solution) > 0 and len(new_solution[0]) > 2:
                route0 = new_solution[0]
                current_load = 0
                for b in route0:
                    if b == 0:
                        continue
                    # Optimization: pre-calculate bin weights? Or just do it here.
                    bin_kg = stocks.get(b, 0) * V * density / 100.0
                    if current_load + bin_kg <= vehicle_capacity:
                        current_load += bin_kg
                        real_kg += bin_kg
                        if must_go_bins and b in must_go_bins:
                            collected_must_go.add(b)
                    else:
                        break
            
            new_revenue = real_kg * R
            # Must-Go Penalty
            missed_must_go = len(must_go_bins) - len(collected_must_go) if must_go_bins else 0
            penalty_must_go = missed_must_go * 10000.0
            
            new_profit = new_revenue - new_cost - penalty_must_go

            delta = new_profit - current_profit
            
            # Acceptance Probability
            if delta > 0:
                accept = True
            else:
                p = math.exp(delta / T)
                accept = (random.random() < p)

            if accept:
                # ACCEPT: Adopt new routes AND the modified removed_bins set
                current_solution = new_solution
                current_cost = new_cost
                current_profit = new_profit
                removed_bins = candidate_removed_bins # <--- CRITICAL: Update state only on acceptance
                
                if current_profit > best_profit:
                    best_solution = copy.deepcopy(current_solution)
                    best_profit = current_profit
                    last_weight = real_kg
                    last_distance = new_cost
                    last_receita = new_revenue
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                # REJECT: Do nothing. 
                # candidate_removed_bins is discarded automatically.
                # removed_bins remains untouched, so "lost" bins are effectively restored.
                no_improvement_count += 1

            T = T * alpha
            if verbose:
                print(f"Temperature cooled to {T:.4f}")

        if no_improvement_count > 500:
            if verbose:
                print("[INFO] Reaquecendo temperatura.")
            T = T_init          # start again from the original temperature
            no_improvement_count = 0

    best_solution = [r for r in best_solution if len(r) > 2]
    best_solution = uncross_arcs_in_sans_routes(best_solution, id_to_index, distance_matrix)
    return best_solution, best_profit, last_distance, last_weight, last_receita
