"""
Operator application logic for Simulated Annealing.
"""

import random

from .sans_neighborhoods import (
    cross_exchange,
    get_2opt_neighbors,
    insert_bin_in_route,
    move_between_routes,
    mutate_route_by_swapping_bins,
    or_opt_move,
    relocate_within_route,
)
from .sans_perturbations import (
    add_n_bins_consecutive,
    add_n_bins_random,
    add_route_with_removed_bins_consecutive,
    add_route_with_removed_bins_random,
    move_n_route_consecutive,
    move_n_route_random,
    remove_n_bins_consecutive,
    remove_n_bins_random,
    swap_n_route_consecutive,
    swap_n_route_random,
)


def apply_operator(
    op,
    new_solution,
    candidate_removed_bins,
    data,
    vehicle_capacity,
    id_to_index,
    stocks,
    must_go_bins,
    distance_matrix,
):
    """
    Apply the selected operator to the solution.
    Modifies new_solution in place or returns a new one.
    """
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
        new_solution = remove_n_bins_random(
            new_solution,
            candidate_removed_bins,
            must_go_bins,
            n=random.randint(2, 5),
        )
    elif op == "remove_n_bins_consec":
        new_solution = remove_n_bins_consecutive(
            new_solution,
            candidate_removed_bins,
            must_go_bins,
            n=random.randint(2, 5),
        )
    elif op == "add_n_bins":
        new_solution = add_n_bins_random(
            new_solution,
            candidate_removed_bins,
            stocks,
            vehicle_capacity,
            id_to_index,
            distance_matrix,
            n=2,
        )
    elif op == "add_n_bins_consec":
        new_solution = add_n_bins_consecutive(
            new_solution,
            candidate_removed_bins,
            stocks,
            vehicle_capacity,
            id_to_index,
            distance_matrix,
            n=2,
        )
    elif op == "add_route_removed":
        new_solution = add_route_with_removed_bins_random(
            new_solution, candidate_removed_bins, stocks, vehicle_capacity
        )
    elif op == "add_route_removed_consec":
        new_solution = add_route_with_removed_bins_consecutive(
            new_solution, candidate_removed_bins, stocks, vehicle_capacity
        )
    elif op == "relocate":
        r = random.choice(range(len(new_solution)))
        new_solution[r] = relocate_within_route(new_solution[r])
    elif op == "cross":
        new_solution = cross_exchange(new_solution)
    elif op == "or-opt":
        r = random.choice(range(len(new_solution)))
        new_solution[r] = or_opt_move(new_solution[r])
    elif op == "insert":
        if new_solution:
            r = random.choice(range(len(new_solution)))
            all_bins = set(data["#bin"]) - {0}
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

    return new_solution
