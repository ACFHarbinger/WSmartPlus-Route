"""
Reversed Search Module.

This module implements a search strategy that explores the solution space
by reversing routes or segments of routes.

Attributes:
    None

Example:
    >>> from logic.src.policies.simulated_annealing_neighborhood_search.search.reversed import ReversedSearch
    >>> search = ReversedSearch(...)
    >>> result = search.run()
"""

from copy import deepcopy

from logic.src.policies.simulated_annealing_neighborhood_search.common.objectives import (
    compute_profit,
    compute_real_profit,
)


def local_search_reversed(
    previous_solution,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Similar to local_search_2 but applies operators in a different order or variant.

    Args:
        previous_solution (List[List[int]]): Solution to improve.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Imbalance penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Bin weights data.
        distance_matrix (np.ndarray): Shortest path matrix.
        values (Dict): Parameter dictionary.

    Returns:
        List[List[int]]: Modified routing solution.
    """
    previous_profit = compute_profit(
        previous_solution,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    routes_list = deepcopy(previous_solution)
    for it, i in reversed(list(enumerate(previous_solution))):
        i = previous_solution[it]
        idx_route = previous_solution.index(i)
        for _j, val in reversed(list(enumerate(i[0 : len(i) - 1]))):
            position = i.index(val)
            row = list(distance_matrix[val][:])
            row_new = sorted(row)

            # Only five closest bins
            e = 51
            c = 51
            for a in row_new[1:e]:
                bin_to_move = row.index(a)

                # Identify route of bin to move
                for z in previous_solution:
                    if bin_to_move in z:
                        index_route_to_remove = previous_solution.index(z)
                        if index_route_to_remove == idx_route:
                            position_bin_to_move = i.index(bin_to_move)
                            place = position if position_bin_to_move < position else position + 1
                        else:
                            place = position + 1
                        if bin_to_move == 0:
                            c = e + 1
                        elif index_route_to_remove == idx_route:
                            routes_list[idx_route].remove(bin_to_move)
                            routes_list[idx_route].insert(place, bin_to_move)
                            new_profit = compute_profit(
                                routes_list,
                                p_vehicle,
                                p_load,
                                p_route_difference,
                                p_shift,
                                data,
                                distance_matrix,
                                values,
                            )
                            if new_profit > previous_profit:
                                previous_solution = deepcopy(routes_list)
                                previous_profit = new_profit
                            else:
                                routes_list = deepcopy(previous_solution)
                        else:
                            routes_list[index_route_to_remove].remove(bin_to_move)
                            routes_list[idx_route].insert(place, bin_to_move)
                            new_profit = compute_profit(
                                routes_list,
                                p_vehicle,
                                p_load,
                                p_route_difference,
                                p_shift,
                                data,
                                distance_matrix,
                                values,
                            )
                            if new_profit > previous_profit:
                                previous_solution = deepcopy(routes_list)
                                previous_profit = new_profit
                            else:
                                routes_list = deepcopy(previous_solution)
            if c > e:
                for b in row_new[e:c]:
                    bin_to_move = row.index(b)
                    for z in previous_solution:
                        if bin_to_move in z:
                            index_route_to_remove = previous_solution.index(z)
                            if index_route_to_remove == idx_route:
                                position_bin_to_move = i.index(bin_to_move)
                                place = position if position_bin_to_move < position else position + 1
                            else:
                                place = position + 1
                            if index_route_to_remove == idx_route:
                                routes_list[idx_route].remove(bin_to_move)
                                routes_list[idx_route].insert(place, bin_to_move)
                                new_profit = compute_profit(
                                    routes_list,
                                    p_vehicle,
                                    p_load,
                                    p_route_difference,
                                    p_shift,
                                    data,
                                    distance_matrix,
                                    values,
                                )
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
                            else:
                                routes_list[index_route_to_remove].remove(bin_to_move)
                                routes_list[idx_route].insert(place, bin_to_move)
                                new_profit = compute_profit(
                                    routes_list,
                                    p_vehicle,
                                    p_load,
                                    p_route_difference,
                                    p_shift,
                                    data,
                                    distance_matrix,
                                    values,
                                )
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
            i = routes_list[idx_route]

    solution_after_local_search_reversed = deepcopy(previous_solution)
    profit_after_local_search_reversed = compute_profit(
        solution_after_local_search_reversed,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    LS_reversed_profit = compute_real_profit(
        solution_after_local_search_reversed, p_vehicle, data, distance_matrix, values
    )
    return (
        solution_after_local_search_reversed,
        profit_after_local_search_reversed,
        LS_reversed_profit,
    )
