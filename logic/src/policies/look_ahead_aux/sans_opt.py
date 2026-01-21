"""
Adaptive mutation and neighborhood operators for SANS-style optimization.

Implements specialized local search operators including cross-exchange,
relocation, Or-opt moves, and 2-opt swaps tailored for the SANS (Simulated
Annealing with Adaptive Neighborhood Selection) policy variant.
"""

import copy
import random

__all__ = [
    "get_neighbors",
    "get_2opt_neighbors",
    "relocate_within_route",
    "cross_exchange",
    "or_opt_move",
    "move_between_routes",
    "insert_bin_in_route",
    "mutate_route_by_swapping_bins",
    "remove_bins_from_route",
    "move_n_route_random",
    "swap_n_route_random",
    "remove_n_bins_random",
    "add_n_bins_random",
    "add_route_with_removed_bins_random",
    "move_n_route_consecutive",
    "swap_n_route_consecutive",
    "remove_n_bins_consecutive",
    "add_n_bins_consecutive",
    "add_route_with_removed_bins_consecutive",
]


def get_neighbors(route):
    """
    Generate the neighborhood of a route through 2-opt swaps.

    Args:
        route (List[int]): Current route.

    Returns:
        List[List[int]]: List of neighboring routes.
    """
    neighbors = []
    if len(route) <= 3:
        return [route]

    for i in range(1, len(route) - 2):
        for j in range(i + 1, len(route) - 1):
            new_route = route[:]
            new_route[i], new_route[j] = new_route[j], new_route[i]
            neighbors.append(new_route)
    return neighbors


def get_2opt_neighbors(route):
    """
    Generate 2-opt neighbors for a route.

    Args:
        route (List[int]): Proposed node sequence.

    Returns:
        List[List[int]]: Candidate neighboring routes.
    """
    neighbors = []
    n = len(route)
    if n <= 4:
        return [route]

    for i in range(1, n - 2):
        for j in range(i + 1, n - 1):
            new_route = route[:i] + route[i : j + 1][::-1] + route[j + 1 :]
            neighbors.append(new_route)
    return neighbors


def relocate_within_route(route):
    """
    Relocate a random bin to a different position in the same route.

    Args:
        route (List[int]): Node sequence.

    Returns:
        List[int]: Mutated route.
    """
    if len(route) <= 3:
        return route[:]
    new_route = route[:]
    i, j = sorted(random.sample(range(1, len(route) - 1), 2))
    bin_moved = new_route.pop(i)
    new_route.insert(j, bin_moved)
    return new_route


def cross_exchange(routes):
    """
    Exchange segments between two different routes.

    Args:
        routes (List[List[int]]): Set of routes.

    Returns:
        List[List[int]]: Mutated solution.
    """
    if len(routes) < 2:
        return routes[:]

    new_routes = copy.deepcopy(routes)
    r1, r2 = random.sample(range(len(new_routes)), 2)
    route1, route2 = new_routes[r1], new_routes[r2]
    if len(route1) <= 3 or len(route2) <= 3:
        return new_routes

    i1 = random.randint(1, len(route1) - 2)
    i2 = random.randint(1, len(route2) - 2)

    new_route1 = route1[:i1] + route2[i2:-1] + [0]
    new_route2 = route2[:i2] + route1[i1:-1] + [0]

    new_routes[r1] = new_route1
    new_routes[r2] = new_route2
    return new_routes


def or_opt_move(route):
    """
    Apply Or-opt operator (move a segment of 1-3 nodes) within a route.

    Args:
        route (List[int]): Node sequence.

    Returns:
        List[int]: Mutated route.
    """
    if len(route) <= 4:
        return route[:]

    new_route = route[:]
    k = random.choice([1, 2])  # tamanho do segmento a ser movido
    start = random.randint(1, len(route) - k - 1)
    segment = new_route[start : start + k]
    del new_route[start : start + k]

    insert_pos = random.randint(1, len(new_route) - 1)
    new_route = new_route[:insert_pos] + segment + new_route[insert_pos:]
    return new_route


def move_between_routes(routes, data, vehicle_capacity, id_to_index):
    """
    Move a random bin from one route to another, respecting capacity constraints.

    Args:
        routes (List[List[int]]): Set of routes.
        data (pd.DataFrame): Bin weights data.
        vehicle_capacity (float): Tanker capacity.
        id_to_index (Dict): Mapping.

    Returns:
        List[List[int]]: Mutated solution.
    """
    moves = []
    stocks = dict(zip(data["#bin"], data["Stock"]))
    for i in range(len(routes)):
        for j in range(len(routes)):
            if i == j:
                continue
            for idx in range(1, len(routes[i]) - 1):
                bin_to_move = routes[i][idx]
                if bin_to_move == 0:
                    continue
                load_j = sum(stocks.get(b, 0) for b in routes[j] if b != 0)
                if load_j + stocks[bin_to_move] > vehicle_capacity:
                    continue
                new_routes = copy.deepcopy(routes)
                new_routes[i].pop(idx)
                insert_pos = random.randint(1, len(new_routes[j]) - 1)
                new_routes[j].insert(insert_pos, bin_to_move)
                moves.append(new_routes)
    return moves


def insert_bin_in_route(route, bin_id, id_to_index, distance_matrix):
    """
    Insert a bin into a route at the position minimizing cost increase.

    Args:
        route (List[int]): Current route.
        bin_id (int): Bin to insert.
        id_to_index (Dict): Mapping from bin ID to matrix index.
        distance_matrix (np.ndarray): Distance matrix.

    Returns:
        List[int]: New route with the inserted bin.
    """
    best_pos = None
    min_increase = float("inf")
    for i in range(1, len(route)):
        prev_bin = route[i - 1]
        next_bin = route[i]
        added_cost = (
            distance_matrix[id_to_index[prev_bin], id_to_index[bin_id]]
            + distance_matrix[id_to_index[bin_id], id_to_index[next_bin]]
            - distance_matrix[id_to_index[prev_bin], id_to_index[next_bin]]
        )
        if added_cost < min_increase:
            min_increase = added_cost
            best_pos = i
    new_route = route[:best_pos] + [bin_id] + route[best_pos:]
    return new_route


def mutate_route_by_swapping_bins(route, num_bins=1):
    """
    Swap random nodes within a route.

    Args:
        route (List[int]): Route to mutate.
        num_bins (int): Number of nodes to swap.

    Returns:
        List[int]: Mutated route.
    """
    new_route = route[:]
    indices = [i for i in range(1, len(route) - 1)]
    if len(indices) < num_bins * 2:
        return new_route
    selected = random.sample(indices, num_bins * 2)
    for i in range(0, len(selected), 2):
        a, b = selected[i], selected[i + 1]
        new_route[a], new_route[b] = new_route[b], new_route[a]
    return new_route


def remove_bins_from_route(route, num_bins=1):
    """
    Remove random bins from a route.

    Args:
        route (List[int]): Route to mutate.
        num_bins (int): Number of bins to remove.

    Returns:
        List[int]: New route with bins removed.
    """
    new_route = route[:]
    indices = [i for i in range(1, len(route) - 1)]
    if len(indices) <= num_bins:
        return new_route
    to_remove = random.sample(indices, num_bins)
    for idx in sorted(to_remove, reverse=True):
        del new_route[idx]
    return new_route


def move_n_route_random(routes_list, n=2):
    """
    Move n random bins from one route to another.

    Args:
        routes_list (List[List[int]]): Current routes.
        n (int): Number of bins to move.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    non_empty = [r for r in new_routes if len(r) > 2]
    if len(non_empty) < 1:
        return new_routes
    r = random.choice(non_empty)
    if len(r) <= n + 2:
        return new_routes
    to_move = random.sample(r[1:-1], n)
    r = [x for x in r if x not in to_move]
    insert_r = random.choice(non_empty)
    insert_pos = random.randint(1, len(insert_r) - 1)
    insert_r = insert_r[:insert_pos] + to_move + insert_r[insert_pos:]
    new_routes = [insert_r if x == insert_r else x for x in new_routes]
    return new_routes


def swap_n_route_random(routes_list, n=2):
    """
    Swap n random bins between two routes.

    Args:
        routes_list (List[List[int]]): Current routes.
        n (int): Number of bins to swap.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    candidates = [r for r in new_routes if len(r) > n + 2]
    if len(candidates) < 2:
        return new_routes
    r1, r2 = random.sample(candidates, 2)
    b1 = random.sample(r1[1:-1], n)
    b2 = random.sample(r2[1:-1], n)
    r1 = [x for x in r1 if x not in b1] + b2
    r2 = [x for x in r2 if x not in b2] + b1
    return [r1 if x == r1 else r2 if x == r2 else x for x in new_routes]


def remove_n_bins_random(routes_list, removed_bins, bins_cannot_removed, n=2):
    """
    Remove n random bins from routes and add to removed set.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (Set): Set of removed bins to update.
        bins_cannot_removed (List[int]): Bins that cannot be removed.
        n (int): Number of bins to remove per route.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    for route in new_routes:
        candidates = [b for b in route[1:-1] if b not in bins_cannot_removed]
        to_remove = random.sample(candidates, min(len(candidates), n))
        for b in to_remove:
            route.remove(b)
            removed_bins.add(b)
    return new_routes


def add_n_bins_random(
    routes_list,
    removed_bins,
    stocks,
    vehicle_capacity,
    id_to_index,
    distance_matrix,
    n=2,
):
    """
    Add n random bins from removed set to routes.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (Set): Set of removed bins.
        stocks (Dict): Bin demand values.
        vehicle_capacity (float): Vehicle capacity.
        id_to_index (Dict): ID to index mapping.
        distance_matrix (np.ndarray): Distance matrix.
        n (int): Number of bins to add.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    if not new_routes:
        # If no routes exist, we can't easily add bins without creating a route
        # For now, just return as is (sa might handle this)
        return new_routes
    bins_to_add = random.sample(list(removed_bins), min(n, len(removed_bins)))
    for b in bins_to_add:
        route = random.choice(new_routes)
        carga = sum(stocks.get(x, 0) for x in route if x != 0)
        if carga + stocks.get(b, 0) <= vehicle_capacity:
            new_route = insert_bin_in_route(route, b, id_to_index, distance_matrix)
            new_routes = [new_route if x == route else x for x in new_routes]
            removed_bins.remove(b)
    return new_routes


def add_route_with_removed_bins_random(routes_list, removed_bins, stocks, vehicle_capacity):
    """
    Create a new route from random removed bins.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (Set): Set of removed bins.
        stocks (Dict): Bin demand values.
        vehicle_capacity (float): Vehicle capacity.

    Returns:
        List[List[int]]: New routes with added route.
    """
    new_routes = copy.deepcopy(routes_list)
    if not removed_bins:
        return new_routes
    new_route = [0]
    carga = 0
    candidates = list(removed_bins)
    random.shuffle(candidates)
    for b in candidates:
        if carga + stocks[b] <= vehicle_capacity:
            new_route.append(b)
            carga += stocks[b]
    new_route.append(0)
    for b in new_route[1:-1]:
        removed_bins.remove(b)
    if len(new_route) > 2:
        new_routes.append(new_route)
    return new_routes


def move_n_route_consecutive(routes_list, n=2):
    """
    Move n consecutive bins from one route to another.

    Args:
        routes_list (List[List[int]]): Current routes.
        n (int): Number of bins to move.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    non_empty = [r for r in new_routes if len(r) > n + 2]
    if not non_empty:
        return new_routes
    r = random.choice(non_empty)
    idx = random.randint(1, len(r) - n - 1)
    subseq = r[idx : idx + n]
    r = r[:idx] + r[idx + n :]
    insert_r = random.choice(non_empty)
    insert_pos = random.randint(1, len(insert_r) - 1)
    insert_r = insert_r[:insert_pos] + subseq + insert_r[insert_pos:]
    return [insert_r if x == insert_r else r if x == r else x for x in new_routes]


def swap_n_route_consecutive(routes_list, n=2):
    """
    Swap n consecutive bins between two routes.

    Args:
        routes_list (List[List[int]]): Current routes.
        n (int): Number of bins to swap.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    valid_routes = [r for r in new_routes if len(r) > n + 2]
    if len(valid_routes) < 2:
        return new_routes
    r1, r2 = random.sample(valid_routes, 2)
    i1 = random.randint(1, len(r1) - n - 1)
    i2 = random.randint(1, len(r2) - n - 1)
    s1 = r1[i1 : i1 + n]
    s2 = r2[i2 : i2 + n]
    r1 = r1[:i1] + s2 + r1[i1 + n :]
    r2 = r2[:i2] + s1 + r2[i2 + n :]
    return [r1 if x == r1 else r2 if x == r2 else x for x in new_routes]


def remove_n_bins_consecutive(routes_list, removed_bins, bins_cannot_removed, n=2):
    """
    Remove n consecutive bins from routes where allowed.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (Set): Set of removed bins.
        bins_cannot_removed (List[int]): Bins that cannot be removed.
        n (int): Number of consecutive bins to remove.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    for route in new_routes:
        indices = [
            i for i in range(1, len(route) - n - 1) if all(route[j] not in bins_cannot_removed for j in range(i, i + n))
        ]
        if not indices:
            continue
        i = random.choice(indices)
        for j in range(n):
            removed_bins.add(route[i])
            del route[i]
    return new_routes


def add_n_bins_consecutive(
    routes_list,
    removed_bins,
    stocks,
    vehicle_capacity,
    id_to_index,
    distance_matrix,
    n=2,
):
    """
    Add n separate bins from removed set to routes using insertion.

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (Set): Set of removed bins.
        stocks (Dict): Bin demand values.
        vehicle_capacity (float): Vehicle capacity.
        id_to_index (Dict): ID to index mapping.
        distance_matrix (np.ndarray): Distance matrix.
        n (int): Number of bins to add.

    Returns:
        List[List[int]]: New routes.
    """
    new_routes = copy.deepcopy(routes_list)
    if not new_routes or len(removed_bins) < n:
        return new_routes
    bins_to_add = random.sample(list(removed_bins), n)
    for route in new_routes:
        carga = sum(stocks.get(b, 0) for b in route if b != 0)
        if carga + sum(stocks[b] for b in bins_to_add) <= vehicle_capacity:
            random.randint(1, len(route) - 1)
            for b in bins_to_add:
                route = insert_bin_in_route(route, b, id_to_index, distance_matrix)
                removed_bins.remove(b)
            break
    return new_routes


def add_route_with_removed_bins_consecutive(routes_list, removed_bins, stocks, vehicle_capacity):
    """
    Create a new route from removed bins (sorted by ID).

    Args:
        routes_list (List[List[int]]): Current routes.
        removed_bins (Set): Set of removed bins.
        stocks (Dict): Bin demand values.
        vehicle_capacity (float): Vehicle capacity.

    Returns:
        List[List[int]]: New routes with added route.
    """
    new_routes = copy.deepcopy(routes_list)
    if len(removed_bins) == 0:
        return new_routes
    sorted_bins = sorted(list(removed_bins))  # ordenação simples
    carga = 0
    nova_rota = [0]
    for b in sorted_bins:
        if carga + stocks.get(b, 0) <= vehicle_capacity:
            nova_rota.append(b)
            carga += stocks[b]
    nova_rota.append(0)
    for b in nova_rota[1:-1]:
        removed_bins.remove(b)
    if len(nova_rota) > 2:
        new_routes.append(nova_rota)
    return new_routes
