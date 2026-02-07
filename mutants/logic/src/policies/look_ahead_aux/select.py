"""
Selection and set-membership logic for route bin management.

Provides utilities for adding and removing bins from the active routing set.
Handles the logic for managing the pool of 'removed' (uncollected) bins,
enabling constructive and destructive perturbations during optimization.
"""

from copy import deepcopy
from random import sample as rsample

from .computations import compute_profit, compute_real_profit
from .routes import organize_route

__all__ = [
    "remove_bin",
    "add_bin",
    "insert_bins",
    "remove_bins_end",
    "remove_n_bins_random",
    "remove_n_bins_consecutive",
    "add_n_bins_random",
    "add_n_bins_consecutive",
    "add_route_random",
    "add_route_consecutive",
    "add_route_with_removed_bins_random",
    "add_route_with_removed_bins_consecutive",
]


# Function to remove one bin from one route (Drop one bin from one random chosen route)
def remove_bin(routes_list, removed_bins, bins_cannot_removed):
    """
    Remove a random bin from one of the routes and add it to the removed set.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Global set of uncollected bins.
        bins_cannot_removed (List[int]): Mandatory bins.

    Returns:
        Tuple[List[List[int]], List[int]]: Updated solution and removed set.
    """
    bin_to_remove = None
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
        if len(chosen_route) > len(bins_cannot_removed) + 2:
            if len(bins_cannot_removed) > 0:
                bin_to_remove = rsample(bins_cannot_removed, 1)[0]
                i = 0
                while bin_to_remove in bins_cannot_removed:  # and i <= len(chosen_route):
                    bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                    i += 1
            else:
                bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

            removed_bins.append(bin_to_remove)
            chosen_route.remove(bin_to_remove)
        if len(chosen_route) == 2:
            routes_list.remove(chosen_route)

        routes_list = list(filter(None, routes_list))
    return bin_to_remove


# Function to add bin from the removed bins set to one route
# (Add one bin from the removed bins set to one random chosen route)
def add_bin(routes_list, removed_bins):
    """
    Pick a random bin from the removed set and add it to a random route.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Set of available removed bins.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    if len(routes_list) > 0:
        if len(removed_bins) > 0:
            bin_to_add = rsample(removed_bins, 1)[0]
            chosen_route = rsample(routes_list, 1)[0]
            chosen_position = chosen_route.index(rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0])

            chosen_route.insert(chosen_position, bin_to_add)
            removed_bins.remove(bin_to_add)
            return bin_to_add


def insert_bins(
    routes_list,
    removed_bins,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Greedily insert bins from the removed set into the current routes if it improves profit.

    Continues until no more profitable insertions are found.

    Args:
        routes_list (List[List[int]]): Initial routes.
        removed_bins (List[int]): Available bins.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Difference penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Weights context.
        distance_matrix (np.ndarray): Distance data.
        values (Dict): Constants.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    inserted_bins = []
    previous_solution = deepcopy(routes_list)
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
    for it, a in enumerate(removed_bins):
        for it1, i in enumerate(previous_solution):
            i = routes_list[it1]
            for it2, j in enumerate(i[0 : len(i) - 1]):
                i = routes_list[it1]
                for b in routes_list:
                    if a in b:
                        b.remove(a)
                position = it2 + 1
                i.insert(position, a)
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
    for c in removed_bins:
        for m in previous_solution:
            if c in m:
                inserted_bins.append(c)

    for ins in inserted_bins:
        removed_bins.remove(ins)

    previous_real_profit = compute_real_profit(previous_solution, p_vehicle, data, distance_matrix, values)
    return previous_solution, previous_profit, previous_real_profit, removed_bins


def remove_bins_end(
    routes_list,
    removed_bins,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    bins_cannot_removed,
    distance_matrix,
    values,
):
    """
    Greedily remove bins from the current routes if it increases total profit.

    Args:
        routes_list (List[List[int]]): Initial routes.
        removed_bins (List[int]): Target set for dropped bins.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Difference penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Bin metrics.
        bins_cannot_removed (List[int]): Mandatory bins.
        distance_matrix (np.ndarray): Metric matrix.
        values (Dict): Constants.

    Returns:
        List[List[int]]: Optimized routing solution.
    """
    previous_solution = deepcopy(routes_list)
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
    for it, i in enumerate(previous_solution):
        i = routes_list[it]
        for j in i[1 : len(i) - 1]:
            i = routes_list[it]
            if j not in bins_cannot_removed:
                i.remove(j)
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
                removed_bins.append(j)
            else:
                routes_list = deepcopy(previous_solution)
    previous_real_profit = compute_real_profit(previous_solution, p_vehicle, data, distance_matrix, values)
    return previous_solution, previous_profit, previous_real_profit, removed_bins


# Function to remove n bin from route randomly
def remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed):
    """
    Remove n random bins from the routing solution.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Target set for removed bins.
        bins_cannot_removed (List[int]): Mandatory bins.

    Returns:
        List[List[int]]: Routing solution with fewer bins.
    """
    bins_to_remove_random = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    length = 0
    for i in routes_list:
        length = length + len(i)

    n_routes = len(routes_list)
    bin_to_remove = None
    if n_routes > 0:
        chosen_route = rsample(routes_list, 1)[0]
        if len(chosen_route) - 2 - len(bins_cannot_removed) > chosen_n:
            for i in range(0, chosen_n):
                if len(bins_cannot_removed) > 0:
                    bin_to_remove = rsample(bins_cannot_removed, 1)[0]
                    i = 0
                    while bin_to_remove in bins_cannot_removed and i <= 100:
                        bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
                        i += 1
                else:
                    bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]

                removed_bins.append(bin_to_remove)
                chosen_route.remove(bin_to_remove)
                bins_to_remove_random.append(bin_to_remove)
            if len(chosen_route) == 2:
                routes_list.remove(chosen_route)
        routes_list = list(filter(None, routes_list))
    return bins_to_remove_random, chosen_n


# Function to remove n bin from route consecutively
def remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed):
    """
    Remove n consecutive bins from a route.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Target set.
        bins_cannot_removed (List[int]): Mandatory nodes.

    Returns:
        List[List[int]]: Mutated routing solution.
    """
    bins_to_remove_consecutive = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]
        n_bins_in_chosen_route_cannot_be_removed = 0
        for x in bins_cannot_removed:
            if x in chosen_route:
                n_bins_in_chosen_route_cannot_be_removed = n_bins_in_chosen_route_cannot_be_removed + 1

        if len(chosen_route) - 2 - n_bins_in_chosen_route_cannot_be_removed > chosen_n:
            if len(bins_cannot_removed) > 0:
                bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - chosen_n - 1], 1)[0]
                position_bin_to_remove_prov = chosen_route.index(bin_to_remove)
                bins = []
                bins.append(bin_to_remove)
                for a in range(1, chosen_n):
                    bin_x = chosen_route[position_bin_to_remove_prov + a]
                    bins.append(bin_x)

                i = 0
                while any(map(lambda each: each in bins, bins_cannot_removed)) and i < 100:
                    bins = []
                    bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - chosen_n - 1], 1)[0]
                    i += 1
                    position_bin_to_remove_prov = chosen_route.index(bin_to_remove)
                    bins.append(bin_to_remove)
                    for a in range(1, chosen_n):
                        bin_x = chosen_route[position_bin_to_remove_prov + a]
                        bins.append(bin_x)
                if not any(map(lambda each: each in bins, bins_cannot_removed)):
                    for c in bins:
                        bins_to_remove_consecutive.append(c)
                        removed_bins.append(c)
            else:
                for j in range(0, chosen_n):
                    if j == 0:
                        bin_to_remove = rsample(chosen_route[1 : len(chosen_route) - chosen_n - 1], 1)[0]
                        position_bin_to_remove = chosen_route.index(bin_to_remove)
                        bins_to_remove_consecutive.append(bin_to_remove)
                        removed_bins.append(bin_to_remove)
                    if j > 0:
                        bin_to_remove = chosen_route[position_bin_to_remove + j]
                        bins_to_remove_consecutive.append(bin_to_remove)
                        removed_bins.append(bin_to_remove)
        for a in bins_to_remove_consecutive:
            chosen_route.remove(a)

        if len(chosen_route) == 2:
            routes_list.remove(chosen_route)

        routes_list = list(filter(None, routes_list))
    return bins_to_remove_consecutive, chosen_n


# Function to add n bin from the removed bins set to any route randomly
def add_n_bins_random(routes_list, removed_bins):
    """
    Add n random bins from the removed set to random routes.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Set of removed bins.

    Returns:
        Tuple[List[int], int]: List of added bins and the number n chosen.
    """
    bins_to_add_random = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    if len(routes_list) > 0:
        if len(removed_bins) >= chosen_n:
            for b in range(0, chosen_n):
                bin_to_add = rsample(removed_bins, 1)[0]
                chosen_route = rsample(routes_list, 1)[0]
                chosen_position = chosen_route.index(rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0])

                chosen_route.insert(chosen_position, bin_to_add)
                removed_bins.remove(bin_to_add)
                bins_to_add_random.append(bin_to_add)
    return bins_to_add_random, chosen_n


# Function to add n bin from the removed bins set to any route randomly
def add_n_bins_consecutive(routes_list, removed_bins):
    """
    Add n consecutive bins from the removed set (treated as list) to a random route.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Set/List of removed bins.

    Returns:
        Tuple[List[int], int]: List of added bins and the number n chosen.
    """
    bins_to_add_consecutive = []
    possible_n = [2, 3, 4, 5]
    chosen_n = rsample(possible_n, 1)[0]
    if len(routes_list) > 0:
        if len(removed_bins) >= chosen_n:
            for b in range(0, chosen_n):
                bin_to_add = rsample(removed_bins, 1)[0]
                removed_bins.remove(bin_to_add)
                bins_to_add_consecutive.append(bin_to_add)

            chosen_route = rsample(routes_list, 1)[0]
            chosen_position = chosen_route.index(rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0])
            for d in range(0, chosen_n):
                chosen_route.insert(chosen_position + d, bins_to_add_consecutive[d])
    return bins_to_add_consecutive, chosen_n


# Add one route random
def add_route_random(routes_list, distance_matrix):
    """
    Add a new route with a single bin chosen randomly from existing routes.

    Args:
        routes_list (List[List[int]]): All routes.
        distance_matrix (np.ndarray): For sequencing.

    Returns:
        List[List[int]]: Updated set of routes.
    """
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]

    length_chosen_route = len(chosen_route)
    possible_percent = [0.3, 0.4, 0.5, 0.6]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []
    for s in range(0, chosen_n_percent):
        chosen_bin = rsample(chosen_route[1 : len(chosen_route) - 1], 1)[0]
        chosen_route.remove(chosen_bin)
        bins.append(chosen_bin)

    route = organize_route(bins, distance_matrix)
    routes_list.append(route)
    if len(chosen_route) == 2:
        chosen_route = []

    routes_list = list(filter(None, routes_list))
    return chosen_n


# Add one route consecutive
def add_route_consecutive(routes_list, distance_matrix):
    """
    Create a new route by extracting a consecutive segment from an existing route.

    Args:
        routes_list (List[List[int]]): Current routes.
        distance_matrix (np.ndarray): Distance matrix for route organization.

    Returns:
        int: The number of bins moved to the new route.
    """
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]

    length_chosen_route = len(chosen_route)
    possible_percent = [0.3, 0.4, 0.5, 0.6]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []
    for s in range(0, chosen_n_percent):
        if s == 0:
            chosen_bin = rsample(chosen_route[1 : len(chosen_route) - chosen_n_percent], 1)[0]
            position_chosen_bin = chosen_route.index(chosen_bin)
        else:
            chosen_bin = chosen_route[position_chosen_bin + s]

        bins.append(chosen_bin)

    chosen_bins = bins.copy()
    route = organize_route(bins, distance_matrix)
    routes_list.append(route)
    for u in chosen_bins:
        chosen_route.remove(u)

    if len(chosen_route) == 2:
        chosen_route = []

    routes_list = list(filter(None, routes_list))
    return chosen_n


# Add one route with bins from removed bins
def add_route_with_removed_bins_random(routes_list, removed_bins, distance_matrix):
    """
    Create a new route using a random subset of removed bins.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Pool of removed bins.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        Tuple[int, List[int]]: Number of bins used and the list of bins used.
    """
    length_removed_bins = len(removed_bins)
    possible_percent = [0.2, 0.3, 0.4, 0.5]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_removed_bins)
    bins_random = []
    for o in range(0, chosen_n_percent):
        chosen_bin = rsample(removed_bins, 1)[0]
        removed_bins.remove(chosen_bin)
        bins_random.append(chosen_bin)

    chosen_bins = bins_random.copy()
    route = organize_route(chosen_bins, distance_matrix)
    routes_list.append(route)
    return chosen_n, bins_random


# Add one route with bins from removed bins consecutive
def add_route_with_removed_bins_consecutive(routes_list, removed_bins, distance_matrix):
    """
    Create a new route using a consecutive sequence of removed bins.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (List[int]): Pool of removed bins.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        Tuple[int, List[int]]: Number of bins used and the list of bins used.
    """
    length_removed_bins = len(removed_bins)
    possible_percent = [0.2, 0.3, 0.4, 0.5]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_removed_bins)
    bins_consecutive = []
    for o in range(0, chosen_n_percent):
        if o == 0:
            chosen_bin = rsample(removed_bins[0 : len(removed_bins) - chosen_n_percent + 1], 1)[0]
            position_chosen_bin = removed_bins.index(chosen_bin)
        else:
            chosen_bin = removed_bins[position_chosen_bin + o]

        bins_consecutive.append(chosen_bin)

    chosen_bins = bins_consecutive.copy()
    chosen_bins_organize = bins_consecutive.copy()
    route = organize_route(chosen_bins_organize, distance_matrix)
    routes_list.append(route)
    for u in chosen_bins:
        removed_bins.remove(u)
    return chosen_n, bins_consecutive
