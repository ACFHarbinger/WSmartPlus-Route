"""
Routing search strategies and orchestration.
Contains the main Simulated Annealing and Local Search loops for the LookAhead policy.
"""

import math
import time
from copy import deepcopy

import numpy as np

from logic.src.policies.look_ahead_aux.check import check_bins_overflowing_feasibility, check_solution_admissibility
from logic.src.policies.look_ahead_aux.computations import compute_profit, compute_real_profit
from logic.src.policies.look_ahead_aux.routes import uncross_arcs_in_routes
from logic.src.policies.look_ahead_aux.search import local_search, local_search_2, local_search_reversed
from logic.src.policies.look_ahead_aux.select import (
    insert_bins,
    remove_bins_end,
)
from logic.src.policies.look_ahead_aux.solution_initialization import find_initial_solution


def find_solutions(
    data,
    bins_coordinates,
    distance_matrix,
    chosen_combination,
    must_go_bins,
    values,
    n_bins,
    points,
    time_limit,
):
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
    initial_solution = find_initial_solution(
        data,
        bins_coordinates,
        distance_matrix,
        n_bins,
        values["vehicle_capacity"],
        values["E"],
        values["B"],
    )
    initial_solution, _, _ = uncross_arcs_in_routes(
        initial_solution,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

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
    initial_sol_profit = compute_profit(
        initial_sol,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    _ = compute_real_profit(initial_sol, p_vehicle, data, distance_matrix, values)
    previous_sol_profit = initial_sol_profit
    routes_list = deepcopy(initial_solution)
    count = 0
    tic = time.perf_counter()
    for i in range(1, number_iterations + 1):
        (
            chosen_procedure,
            _,
            bin_to_remove,
            bin_to_add,
            bins_to_remove_random,
            bins_to_remove_consecutive,
            bins_to_add_random,
            bins_to_add_consecutive,
            bins_random,
            bins_consecutive,
        ) = local_search(routes_list, removed_bins, distance_matrix, must_go_bins)
        count = count + 1

        # Compute profit for the new solution
        current_sol_profit = compute_profit(
            routes_list,
            p_vehicle,
            p_load,
            p_route_difference,
            p_shift,
            data,
            distance_matrix,
            values,
        )

        # Check candidate route feasibility
        status = check_bins_overflowing_feasibility(
            data,
            routes_list,
            n_bins,
            values["perc_bins_can_overflow"],
            values["E"],
            values["B"],
        )

        # Check candidate route admissibility
        _ = check_solution_admissibility(routes_list, removed_bins, n_bins)
        if status == "pass":
            # Compare current profit with previous profit
            delta = current_sol_profit - previous_sol_profit
            if delta > 0:
                previous_sol = deepcopy(routes_list)
                previous_sol_profit = current_sol_profit
            elif delta == 0:
                if chosen_procedure == "Drop bin" and bin_to_remove is not None:
                    removed_bins.remove(bin_to_remove)
                if chosen_procedure == "Add bin" and bin_to_add is not None:
                    removed_bins.append(bin_to_add)
                if chosen_procedure == "Remove n bins random" and bins_to_remove_random != []:
                    for n in bins_to_remove_random:
                        removed_bins.remove(n)
                if chosen_procedure == "Remove n bins consecutive" and bins_to_remove_consecutive != []:
                    for o in bins_to_remove_consecutive:
                        removed_bins.remove(o)
                if chosen_procedure == "Add n bins random" and bins_to_add_random != []:
                    for r in bins_to_add_random:
                        removed_bins.append(r)
                if chosen_procedure == "Add n bins consecutive" and bins_to_add_consecutive != []:
                    for s in bins_to_add_consecutive:
                        removed_bins.append(s)
                if chosen_procedure == "Add route with removed bins random":
                    for t in bins_random:
                        removed_bins.append(t)
                if chosen_procedure == "Add route with removed bins consecutive":
                    for u in bins_consecutive:
                        removed_bins.append(u)
            else:
                T = T_initial / (i**T_param)
                p = math.exp(delta / T)
                p_random = np.random.uniform(low=0.0, high=1.0, size=None)
                if p >= p_random:
                    previous_sol = deepcopy(routes_list)
                    previous_sol_profit = current_sol_profit
                else:
                    if chosen_procedure == "Drop bin" and bin_to_remove is not None:
                        removed_bins.remove(bin_to_remove)
                    if chosen_procedure == "Add bin" and bin_to_add is not None:
                        removed_bins.append(bin_to_add)
                    if chosen_procedure == "Remove n bins random" and bins_to_remove_random != []:
                        for n in bins_to_remove_random:
                            removed_bins.remove(n)
                        bins_to_remove_random = []
                    if chosen_procedure == "Remove n bins consecutive" and bins_to_remove_consecutive != []:
                        for o in bins_to_remove_consecutive:
                            removed_bins.remove(o)
                    if chosen_procedure == "Add n bins random" and bins_to_add_random != []:
                        for r in bins_to_add_random:
                            removed_bins.append(r)
                    if chosen_procedure == "Add n bins consecutive" and bins_to_add_consecutive != []:
                        for s in bins_to_add_consecutive:
                            removed_bins.append(s)
                    if chosen_procedure == "Add route with removed bins random":
                        for t in bins_random:
                            removed_bins.append(t)
                    if chosen_procedure == "Add route with removed bins consecutive":
                        for u in bins_consecutive:
                            removed_bins.append(u)
        if status == "fail":
            if chosen_procedure == "Drop bin" and bin_to_remove is not None:
                removed_bins.remove(bin_to_remove)
            if chosen_procedure == "Add bin" and bin_to_add is not None:
                removed_bins.append(bin_to_add)
            if chosen_procedure == "Remove n bins random" and bins_to_remove_random != []:
                for n in bins_to_remove_random:
                    removed_bins.remove(n)
            if chosen_procedure == "Remove n bins consecutive" and bins_to_remove_consecutive != []:
                for o in bins_to_remove_consecutive:
                    removed_bins.remove(o)
            if chosen_procedure == "Add n bins random" and bins_to_add_random != []:
                for r in bins_to_add_random:
                    removed_bins.append(r)
            if chosen_procedure == "Add n bins consecutive" and bins_to_add_consecutive != []:
                for s in bins_to_add_consecutive:
                    removed_bins.append(s)
            if chosen_procedure == "Add route with removed bins random":
                for t in bins_random:
                    removed_bins.append(t)
            if chosen_procedure == "Add route with removed bins consecutive":
                for u in bins_consecutive:
                    removed_bins.append(u)

        routes_list = deepcopy(previous_sol)
        _ = compute_real_profit(previous_sol, p_vehicle, data, distance_matrix, values)
        if (time.perf_counter() - tic) > time_limit:
            break

    SA_sol = deepcopy(previous_sol)
    _ = compute_real_profit(SA_sol, p_vehicle, data, distance_matrix, values)
    # end SA

    # Uncross after SA
    solution_after_uncross_1, _, _ = uncross_arcs_in_routes(
        SA_sol,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # First local search
    solution_after_LS_1, _, _ = local_search_2(
        solution_after_uncross_1,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after first local search
    solution_after_uncross_2, _, _ = uncross_arcs_in_routes(
        solution_after_LS_1,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # First reversed local search
    solution_after_reversed_LS_1, _, _ = local_search_reversed(
        solution_after_uncross_2,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after first reversed local search
    solution_after_uncross_3, _, _ = uncross_arcs_in_routes(
        solution_after_reversed_LS_1,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Second local search
    solution_after_LS_2, _, _ = local_search_2(
        solution_after_uncross_3,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after second local search
    solution_after_uncross_4, _, _ = uncross_arcs_in_routes(
        solution_after_LS_2,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Second reversed local search
    solution_after_reversed_LS_2, _, _ = local_search_reversed(
        solution_after_uncross_4,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after second reversed local search
    solution_after_uncross_5, _, _ = uncross_arcs_in_routes(
        solution_after_reversed_LS_2,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Third local search
    solution_after_LS_3, _, _ = local_search_2(
        solution_after_uncross_5,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after third local search
    solution_after_uncross_6, _, _ = uncross_arcs_in_routes(
        solution_after_LS_3,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Third reversed local search
    solution_after_reversed_LS_3, _, _ = local_search_reversed(
        solution_after_uncross_6,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after third reversed local search
    solution_after_uncross_7, _, __ = uncross_arcs_in_routes(
        solution_after_reversed_LS_3,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Fourth local search
    solution_after_LS_4, _, _ = local_search_2(
        solution_after_uncross_7,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after fourth local search
    solution_after_uncross_8, _, _ = uncross_arcs_in_routes(
        solution_after_LS_4,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Fourth reversed local search
    solution_after_reversed_LS_4, _, _ = local_search_reversed(
        solution_after_uncross_8,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after fourth reversed local search
    solution_after_uncross_9, _, _ = uncross_arcs_in_routes(
        solution_after_reversed_LS_4,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Fifth local search
    solution_after_LS_5, _, _ = local_search_2(
        solution_after_uncross_9,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after fifth local search
    solution_after_uncross_10, _, _ = uncross_arcs_in_routes(
        solution_after_LS_5,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Fifth reversed local search
    solution_after_reversed_LS_5, _, _ = local_search_reversed(
        solution_after_uncross_10,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Uncross after fifth reversed local search
    solution_after_uncross_11, _, _ = uncross_arcs_in_routes(
        solution_after_reversed_LS_5,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        points,
        distance_matrix,
        values,
    )

    # Remove bins local search - 1
    solution_after_remove, _, _, removed_bins = remove_bins_end(
        solution_after_uncross_11,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 1
    solution_after_insert, _, _, removed_bins = insert_bins(
        solution_after_remove,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 2
    solution_after_remove_2, _, _, removed_bins = remove_bins_end(
        solution_after_insert,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 2
    solution_after_insert_2, _, _, removed_bins = insert_bins(
        solution_after_remove_2,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 3
    solution_after_remove_3, _, _, removed_bins = remove_bins_end(
        solution_after_insert_2,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 3
    solution_after_insert_3, _, _, removed_bins = insert_bins(
        solution_after_remove_3,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 4
    solution_after_remove_4, _, _, removed_bins = remove_bins_end(
        solution_after_insert_3,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 4
    solution_after_insert_4, _, _, removed_bins = insert_bins(
        solution_after_remove_4,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 5
    solution_after_remove_5, _, _, removed_bins = remove_bins_end(
        solution_after_insert_4,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 5
    solution_after_insert_5, _, _, removed_bins = insert_bins(
        solution_after_remove_5,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local6
    solution_after_remove_6, _, _, removed_bins = remove_bins_end(
        solution_after_insert_5,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 6
    solution_after_insert_6, _, _, removed_bins = insert_bins(
        solution_after_remove_6,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 7
    solution_after_remove_7, _, _, removed_bins = remove_bins_end(
        solution_after_insert_6,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 7
    solution_after_insert_7, _, _, removed_bins = insert_bins(
        solution_after_remove_7,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 8
    solution_after_remove_8, _, _, removed_bins = remove_bins_end(
        solution_after_insert_7,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 8
    solution_after_insert_8, _, _, removed_bins = insert_bins(
        solution_after_remove_8,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 9
    solution_after_remove_9, _, _, removed_bins = remove_bins_end(
        solution_after_insert_8,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 9
    solution_after_insert_9, _, _, removed_bins = insert_bins(
        solution_after_remove_9,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    # Remove bins local search - 10
    solution_after_remove_10, _, _, removed_bins = remove_bins_end(
        solution_after_insert_9,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        must_go_bins,
        distance_matrix,
        values,
    )

    # Insert bins local search - 10
    (
        solution_after_insert_10,
        _,
        real_profit_after_insert_10,
        removed_bins,
    ) = insert_bins(
        solution_after_remove_10,
        removed_bins,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )

    routes = solution_after_insert_10
    profit = real_profit_after_insert_10
    return routes, profit, removed_bins
