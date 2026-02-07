"""
Structural routing utilities and geometric optimization routines.

Handles the creation of spatial coordinate mappings and implements geometric
improvements such as arc uncrossing using LineString intersections. Includes
logic for route reorganization and greedy sequencing.
"""

from copy import deepcopy
from random import sample as rsample

from shapely.geometry import LineString

from .distance import compute_sans_route_cost
from .objectives import compute_profit, compute_real_profit


def create_points(data, bins_coordinates):
    """
    Create a mapping of bin IDs to their coordinates.

    Args:
        data (pd.DataFrame): Bin metadata containing '#bin'.
        bins_coordinates (List[Tuple[float, float]]): List of coordinate pairs.

    Returns:
        Dict[int, Tuple[float, float]]: Mapping from bin ID to coordinates.
    """
    points = []
    pair_lng_lat = []
    for index, row in data.iterrows():
        lat = bins_coordinates["Lat"][index]
        lng = bins_coordinates["Lng"][index]
        pair_lng_lat = []
        pair_lng_lat.append(lng)
        pair_lng_lat.append(lat)
        points.append(pair_lng_lat)
    return points


def uncross_arcs_in_routes(
    previous_solution,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    points,
    distance_matrix,
    values,
):
    """
    Improve routes by uncrossing arcs while maintaining profit constraints.

    Uses geometric LineString intersections to identify crossed arcs.

    Args:
        previous_solution (List[List[int]]): Solution to improve.
        p_vehicle (float): Vehicle penalty.
        p_load (float): Load penalty.
        p_route_difference (float): Difference penalty.
        p_shift (float): Shift penalty.
        data (pd.DataFrame): Bin weights data.
        points (Dict): Bin coordinates.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary.

    Returns:
        List[List[int]]: Improved routing solution.
    """
    arc_pair = []
    arc1 = []
    arc2 = []
    for k in previous_solution:
        index_route = previous_solution.index(k)
        globals()["crossed_arcs_route_{0}".format(index_route)] = []
        for i, val1 in enumerate(k[0 : len(k) - 3]):
            for j, val2 in enumerate(k[i + 2 : len(k) - 1]):
                I_1 = val1
                F_1 = k[i + 1]
                I_2 = k[j + i + 2]
                F_2 = k[j + i + 3]
                line1 = LineString([(points[I_1][0], points[I_1][1]), (points[F_1][0], points[F_1][1])])
                line2 = LineString([(points[I_2][0], points[I_2][1]), (points[F_2][0], points[F_2][1])])
                if line1.intersects(line2):
                    arc1.append(I_1)
                    arc1.append(F_1)
                    arc2.append(I_2)
                    arc2.append(F_2)
                    arc_pair.append(arc1)
                    arc_pair.append(arc2)
                    arc1 = []
                    arc2 = []
                    globals()["crossed_arcs_route_{0}".format(index_route)].append(arc_pair)
                    arc_pair = []

        globals()["relevant_crossed_arcs_route_{0}".format(index_route)] = deepcopy(
            globals()["crossed_arcs_route_{0}".format(index_route)]
        )
        for i in globals()["relevant_crossed_arcs_route_{0}".format(index_route)]:
            if 0 in i[0] and 0 in i[1]:
                globals()["relevant_crossed_arcs_route_{0}".format(index_route)].remove(i)

        for x in globals()["relevant_crossed_arcs_route_{0}".format(index_route)]:
            if points[x[0][1]][0] == points[x[1][0]][0] and points[x[0][1]][1] == points[x[1][0]][1]:
                globals()["relevant_crossed_arcs_route_{0}".format(index_route)].remove(x)

        while len(globals()["relevant_crossed_arcs_route_{0}".format(index_route)]) != 0:
            globals()["crossed_arcs_route_{0}".format(index_route)] = []
            for i, val1 in enumerate(k[0 : len(k) - 3]):
                for j, val2 in enumerate(k[i + 2 : len(k) - 1]):
                    I_1 = val1
                    F_1 = k[i + 1]
                    I_2 = k[j + i + 2]
                    F_2 = k[j + i + 3]
                    line1 = LineString(
                        [
                            (points[I_1][0], points[I_1][1]),
                            (points[F_1][0], points[F_1][1]),
                        ]
                    )
                    line2 = LineString(
                        [
                            (points[I_2][0], points[I_2][1]),
                            (points[F_2][0], points[F_2][1]),
                        ]
                    )
                    if line1.intersects(line2):
                        arc1.append(I_1)
                        arc1.append(F_1)
                        arc2.append(I_2)
                        arc2.append(F_2)
                        arc_pair.append(arc1)
                        arc_pair.append(arc2)
                        arc1 = []
                        arc2 = []
                        globals()["crossed_arcs_route_{0}".format(index_route)].append(arc_pair)
                        arc_pair = []

            globals()["relevant_crossed_arcs_route_{0}".format(index_route)] = deepcopy(
                globals()["crossed_arcs_route_{0}".format(index_route)]
            )
            for i in globals()["relevant_crossed_arcs_route_{0}".format(index_route)]:
                if 0 in i[0] and 0 in i[1]:
                    globals()["relevant_crossed_arcs_route_{0}".format(index_route)].remove(i)

            for x in globals()["relevant_crossed_arcs_route_{0}".format(index_route)]:
                if points[x[0][1]][0] == points[x[1][0]][0] and points[x[0][1]][1] == points[x[1][0]][1]:
                    globals()["relevant_crossed_arcs_route_{0}".format(index_route)].remove(x)

            if len(globals()["relevant_crossed_arcs_route_{0}".format(index_route)]) > 0:
                swap_bin_1 = globals()["relevant_crossed_arcs_route_{0}".format(index_route)][0][0][1]
                position_bin_1 = k.index(swap_bin_1)
                swap_bin_2 = globals()["relevant_crossed_arcs_route_{0}".format(index_route)][0][1][0]
                position_bin_2 = k.index(swap_bin_2)

                k[position_bin_1], k[position_bin_2] = (
                    k[position_bin_2],
                    k[position_bin_1],
                )
                k[position_bin_1 + 1 : position_bin_2] = k[position_bin_2 - 1 : position_bin_1 : -1]

    solution_after_uncross = deepcopy(previous_solution)
    profit_after_uncross = compute_profit(
        solution_after_uncross,
        p_vehicle,
        p_load,
        p_route_difference,
        p_shift,
        data,
        distance_matrix,
        values,
    )
    uncross_profit = compute_real_profit(solution_after_uncross, p_vehicle, data, distance_matrix, values)
    return solution_after_uncross, profit_after_uncross, uncross_profit


# Rearrange part of the route with nearest neighbor rule
def rearrange_part_route(routes_list, distance_matrix):
    """
    Select a random portion of a route and reorder it using a greedy Nearest Neighbor heuristic.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        List[List[int]]: Routing solution with partially reordered route.
    """
    if len(routes_list) > 0:
        chosen_route = rsample(routes_list, 1)[0]

    length_chosen_route = len(chosen_route)
    possible_percent = [0.1, 0.2, 0.3, 0.4]
    chosen_n = rsample(possible_percent, 1)[0]
    chosen_n_percent = int(chosen_n * length_chosen_route)
    bins = []
    for s in range(0, chosen_n_percent):
        if s == 0:
            chosen_bin_0 = rsample(chosen_route[1 : len(chosen_route) - chosen_n_percent], 1)[0]
            position_chosen_bin = chosen_route.index(chosen_bin_0)
        elif s <= chosen_n_percent - 1:
            chosen_bin = chosen_route[position_chosen_bin + s]
            bins.append(chosen_bin)
        else:
            chosen_route[position_chosen_bin + s]

    chosen_bins = bins.copy()
    route = organize_route(bins, distance_matrix)
    route.remove(0)
    route.remove(0)
    for i in chosen_bins:
        chosen_route.remove(i)

    a = 0
    for y in route:
        a = a + 1
        chosen_route.insert(position_chosen_bin + a, y)
    return chosen_n


# Function to generate a route using the nearest neighbor (starting from the left)
def organize_route(bins, distance_matrix):
    """
    Construct a route from a set of bins using the Nearest Neighbor heuristic.

    Starts from the first bin (depot) and iteratively adds the closest unvisited bin.

    Args:
        bins (List[int]): Set of bin IDs to sequence.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        List[int]: Sequenced route starting and ending at the depot.
    """
    depot = 0
    route = []
    distances = []
    bins_sequence = []

    # Choose depot to initialize the route
    bin_chosen = depot
    route.append(bin_chosen)

    # Previous_bin is the bin previously added in the route
    previous_bin = bin_chosen
    while len(bins) != 0:
        for i in bins:
            dist = distance_matrix[previous_bin][i]
            distances.append(dist)
            bins_sequence.append(i)

        min_distance = min(distances)
        distance_index = distances.index(min_distance)
        bin_chosen = bins_sequence[distance_index]
        route.append(bin_chosen)
        previous_bin = bin_chosen

        bins.remove(previous_bin)
        distances = []
        bins_sequence = []
    route.append(depot)
    return route


# Lookahead sans policy functions
def two_opt_uncross_arc(route, distance_matrix, id_to_index):
    """
    Apply 2-opt local search to a single route to remove arc crossings.

    Args:
        route (List[int]): Node sequence.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict[int, int]): Node ID to matrix index mapping.

    Returns:
        List[int]: Optimized route.
    """
    best = route
    improved = True
    while improved:
        improved = False
        best_distance = compute_sans_route_cost(best, distance_matrix, id_to_index)
        for i in range(1, len(best) - 2):
            for j in range(i + 1, len(best) - 1):
                new_route = best[:i] + best[i : j + 1][::-1] + best[j + 1 :]
                new_distance = compute_sans_route_cost(new_route, distance_matrix, id_to_index)
                if new_distance < best_distance:
                    best = new_route
                    improved = True
                    break
            if improved:
                break
    return best


def uncross_arcs_in_sans_routes(routes, id_to_index, distance_matrix):
    """
    Apply 2-opt uncrossing to all routes in a solution.

    Args:
        routes (List[List[int]]): Solution to optimize.
        id_to_index (Dict[int, int]): Node mapping.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        List[List[int]]: Improved solution.
    """
    cleaned_routes = [[int(x) for x in route] for route in routes]
    new_routes = [
        (two_opt_uncross_arc(route, distance_matrix, id_to_index) if len(route) > 3 else route)
        for route in cleaned_routes
    ]
    return new_routes
