# Functions for local search
import numpy as np

from copy import deepcopy
from .move import *
from .swap import *
from .select import *
from .routes import rearrange_part_route
from .computations import compute_profit, compute_real_profit


# Strategy to choose procedure randomly
def local_search(routes_list, removed_bins, distance_matrix, bins_cannot_removed):
    procedures = ['Move 1 route', 'Swap 1 route', 'Move 2 routes', 'Swap 2 routes', 'Drop bin', 'Add bin', \
                  'Move n 1 route random', 'Move n 1 route consecutive', 'Swap n 1 route random', 'Swap n 1 route consecutive', \
                  'Move n 2 routes random', 'Move n 2 routes consecutive', 'Swap n 2 routes random', 'Swap n 2 routes consecutive', \
                  'Remove n bins random', 'Remove n bins consecutive', 'Add n bins random', 'Add n bins consecutive', \
                  'Add route random', 'Add route consecutive', 'Add route with removed bins random', \
                  'Add route with removed bins consecutive', 'Rearrange part of 1 route'] #'Remove bins' #'Insert bins'

    chosen_procedure = np.random.choice(procedures, 1, p=[
        1/19+4.440892098500626e-16,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,1/19,0,0,0,0,1/19
    ])[0]
    bin_to_remove = None
    bin_to_add = None
    bins_to_remove_random = []
    bins_to_remove_consecutive = []
    bins_to_add_random = []
    bins_to_add_consecutive = []
    bins_random = []
    bins_consecutive = []
    chosen_n = 1
    if chosen_procedure == 'Move 1 route':
        move_1_route(routes_list)
    elif chosen_procedure == 'Swap 1 route':
        swap_1_route(routes_list)
    elif chosen_procedure == 'Move 2 routes':
        move_2_routes(routes_list)
    elif chosen_procedure == 'Swap 2 routes':
        swap_2_routes(routes_list)
    elif chosen_procedure == 'Drop bin':
        bin_to_remove = remove_bin(routes_list, removed_bins, bins_cannot_removed)
    elif chosen_procedure == 'Add bin':
        bin_to_add = add_bin(routes_list, removed_bins)
    elif chosen_procedure == 'Move n 1 route random':
        chosen_n = move_n_route_random(routes_list)
    elif chosen_procedure == 'Move n 1 route consecutive':
        chosen_n = move_n_route_consecutive(routes_list)
    elif chosen_procedure == 'Swap n 1 route random':
        chosen_n = swap_n_route_random(routes_list)
    elif chosen_procedure == 'Swap n 1 route consecutive':
        chosen_n = swap_n_route_consecutive(routes_list)
    elif chosen_procedure == 'Move n 2 routes random':
        chosen_n = move_n_2_routes_random(routes_list)
    elif chosen_procedure == 'Move n 2 routes consecutive':
        chosen_n = move_n_2_routes_consecutive(routes_list)
    elif chosen_procedure == 'Swap n 2 routes random':
        chosen_n = swap_n_2_routes_random(routes_list)
    elif chosen_procedure == 'Swap n 2 routes consecutive':
        chosen_n = swap_n_2_routes_consecutive(routes_list)
    elif chosen_procedure == 'Remove n bins random':
        bins_to_remove_random,chosen_n = remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed)
    elif chosen_procedure == 'Remove n bins consecutive':
        bins_to_remove_consecutive,chosen_n = remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed)
    elif chosen_procedure == 'Add n bins random':
        bins_to_add_random,chosen_n = add_n_bins_random(routes_list, removed_bins)
    elif chosen_procedure == 'Add n bins consecutive':
        bins_to_add_consecutive,chosen_n = add_n_bins_consecutive(routes_list, removed_bins)
    elif chosen_procedure == 'Add route random':
        chosen_n = add_route_random(routes_list, distance_matrix)
    elif chosen_procedure == 'Add route consecutive':
        chosen_n = add_route_consecutive(routes_list, distance_matrix)
    elif chosen_procedure == 'Add route with removed bins random':
        chosen_n,bins_random = add_route_with_removed_bins_random(routes_list,removed_bins,distance_matrix)
    elif chosen_procedure == 'Add route with removed bins consecutive':
        chosen_n,bins_consecutive = add_route_with_removed_bins_consecutive(routes_list,removed_bins,distance_matrix)
    else:
        chosen_n = rearrange_part_route(routes_list, distance_matrix)
    return chosen_procedure, chosen_n, bin_to_remove, bin_to_add, bins_to_remove_random, bins_to_remove_consecutive, bins_to_add_random, bins_to_add_consecutive, bins_random, bins_consecutive


def local_search_2(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values):
    previous_profit = compute_profit(previous_solution, p_vehicle, p_load,p_route_difference, p_shift, data, distance_matrix, values)
    routes_list = deepcopy(previous_solution)
    for it, i in enumerate(previous_solution):
        i = previous_solution[it]
        idx_route = previous_solution.index(i)
        for j, val in enumerate(i[0:len(i)-1]):
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
                        route_to_remove = z
                        index_route_to_remove = previous_solution.index(z)
                        if index_route_to_remove == idx_route:
                            position_bin_to_move = i.index(bin_to_move)
                            if position_bin_to_move < position:
                                place = position
                            else:
                                place = position+1
                        else:
                            place = position + 1
                        if bin_to_move == 0:
                            c = e + 1
                        else:
                            if index_route_to_remove == idx_route:
                                routes_list[idx_route].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
                            else:
                                routes_list[index_route_to_remove].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
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
                            route_to_remove = z
                            index_route_to_remove = previous_solution.index(z)
                            if index_route_to_remove == idx_route:
                                position_bin_to_move = i.index(bin_to_move)
                                if position_bin_to_move < position:
                                    place = position
                                else:
                                    place = position+1
                            else:
                                place = position + 1
                            if index_route_to_remove == idx_route:
                                routes_list[idx_route].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
                            else:
                                routes_list[index_route_to_remove].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
            i = routes_list[idx_route]

    solution_after_local_search = deepcopy(previous_solution)
    profit_after_local_search = compute_profit(solution_after_local_search, p_vehicle,p_load, p_route_difference,p_shift, data, distance_matrix, values)
    LS_profit = compute_real_profit(solution_after_local_search, p_vehicle, data, distance_matrix, values)
    return solution_after_local_search,profit_after_local_search,LS_profit


def local_search_reversed(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values):
    previous_profit = compute_profit(previous_solution, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
    routes_list = deepcopy(previous_solution)
    for it, i in reversed(list(enumerate(previous_solution))):
        i = previous_solution[it]
        idx_route = previous_solution.index(i)
        for j, val in reversed(list(enumerate(i[0:len(i)-1]))):
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
                        route_to_remove = z
                        index_route_to_remove = previous_solution.index(z)
                        if index_route_to_remove == idx_route:
                            position_bin_to_move = i.index(bin_to_move)
                            if position_bin_to_move < position:
                                place = position
                            else:
                                place = position+1
                        else:
                            place = position + 1
                        if bin_to_move == 0:
                            c = e + 1
                        else:
                            if index_route_to_remove == idx_route:
                                routes_list[idx_route].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
                            else:
                                routes_list[index_route_to_remove].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
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
                            route_to_remove = z
                            index_route_to_remove = previous_solution.index(z)
                            if index_route_to_remove == idx_route:
                                position_bin_to_move = i.index(bin_to_move)
                                if position_bin_to_move < position:
                                    place = position
                                else:
                                    place = position+1
                            else:
                                place = position + 1
                            if index_route_to_remove == idx_route:
                                routes_list[idx_route].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
                            else:
                                routes_list[index_route_to_remove].remove(bin_to_move)
                                routes_list[idx_route].insert(place,bin_to_move)
                                new_profit = compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
                                if new_profit > previous_profit:
                                    previous_solution = deepcopy(routes_list)
                                    previous_profit = new_profit
                                else:
                                    routes_list = deepcopy(previous_solution)
            i = routes_list[idx_route]

    solution_after_local_search_reversed = deepcopy(previous_solution)
    profit_after_local_search_reversed = compute_profit(solution_after_local_search_reversed, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values)
    LS_reversed_profit = compute_real_profit(solution_after_local_search_reversed, p_vehicle, data, distance_matrix, values)
    return solution_after_local_search_reversed,profit_after_local_search_reversed,LS_reversed_profit