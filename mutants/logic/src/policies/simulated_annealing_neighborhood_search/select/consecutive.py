"""
Consecutive selection strategies for bin management.
"""

from random import sample as rsample

from logic.src.policies.simulated_annealing_neighborhood_search.common.routes import organize_route


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
