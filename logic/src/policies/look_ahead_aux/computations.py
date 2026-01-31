"""
Mathematical and economic performance metrics for greedy routing policies.

Provides utilities for calculating deterministic and stochastic components of
the objective function, including revenue from waste collection, operational
transportation costs, and various penalties (fleet size, vehicle load,
schedule imbalance, and shift duration).
"""


from logic.src.constants.optimization import (
    COLLECTION_TIME_MINUTES,
    MAX_CAPACITY_PERCENT,
    VEHICLE_SPEED_KMH,
)


# Functions for computations
# Computation of profit - Objective function result
# function to compute waste total revenue
def compute_waste_collection_revenue(routes_list, data, E, B, R):
    """
    Calculate the total revenue from waste collected in the current routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        data (pd.DataFrame): Bin data (Stock and Accum_Rate).
        E (float): Bin volume.
        B (float): Bin density.
        R (float): Revenue per kg.

    Returns:
        float: Total collection revenue.
    """
    total_revenue_route = 0
    total_revenue = 0
    for i in routes_list:
        for j in i:
            stock = data["Stock"][j]
            accumulation_rate = data["Accum_Rate"][j]
            bin_stock = stock + accumulation_rate
            revenue_per_bin = bin_stock * E * B * R
            total_revenue_route += revenue_per_bin
        total_revenue += total_revenue_route
        total_revenue_route = 0
    return total_revenue


# Computation of profit - Objective function result
# function to compute distance travelled per route
def compute_distance_per_route(routes_list, distance_matrix):
    """
    Calculate the total travel distance for each individual route.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_matrix (np.ndarray): All-pairs shortest path distances.

    Returns:
        List[float]: Distance travelled by each vehicle/route.
    """
    travelled_distance_route = 0
    distance_route_vector = []
    for i in routes_list:
        for j, element in enumerate(i):
            if j < len(i) - 1:
                travelled_distance_route += distance_matrix[i[j]][i[j + 1]]
        distance_route_vector.append(travelled_distance_route)
        travelled_distance_route = 0
    return distance_route_vector


# Computation of route time
# each bin in the route takes 3 minutes to be collected
# speed - 40km/h
def compute_route_time(routes_list, distance_route_vector):
    """
    Estimate total duration of each route (collection time + travel time).

    Assumes:
    - COLLECTION_TIME_MINUTES minutes per bin collection.
    - VEHICLE_SPEED_KMH km/h average vehicle speed.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_route_vector (List[float]): Distances per route.

    Returns:
        List[float]: Duration in minutes for each route.
    """
    route_time_vector = []
    for route_n, i in enumerate(routes_list):
        route_time = (len(i) - 2) * COLLECTION_TIME_MINUTES + (
            (distance_route_vector[route_n] / VEHICLE_SPEED_KMH) * 60
        )
        route_time_vector.append(route_time)
    return route_time_vector


# Computation of profit - Objective function result
# function to compute transportation cost and distance per route
def compute_transportation_cost(routes_list, distance_route_vector, C):
    """
    Calculate the total transportation cost based on distance and cost coefficient.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        distance_route_vector (List[float]): Distances per route.
        C (float): Cost per distance unit.

    Returns:
        float: Total transportation cost.
    """
    total_travelled_distance = sum(distance_route_vector)
    transportation_cost = C * total_travelled_distance
    return transportation_cost


# Computation of profit - Objective function result
# function to compute vehicle use penalty
# each route corresponds to one vehicle
def compute_vehicle_use_penalty(routes_list, p_vehicle):
    """
    Calculate the penalty for the number of vehicles (routes) used.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Penalty coefficient per vehicle.

    Returns:
        float: Total vehicle penalty.
    """
    number_vehicles = len(routes_list)
    vehicle_penalty = p_vehicle * number_vehicles
    return vehicle_penalty


# Computation of profit - Objective function result
# function to compute difference between longest and shortest route length penalty
def compute_route_time_difference_penalty(routes_list, p_route_difference, distance_route_vector):
    """
    Calculate penalty for workload imbalance between routes.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_route_difference (float): Penalty coefficient for time difference.
        distance_route_vector (List[float]): Distances per route.

    Returns:
        float: Calculated imbalance penalty.
    """
    route_time_vector = compute_route_time(routes_list, distance_route_vector)
    if len(route_time_vector) >= 1:
        min_route_time = min(route_time_vector)
        max_route_time = max(route_time_vector)
        length_difference = max_route_time - min_route_time
        route_difference_penalty = p_route_difference * length_difference
    else:
        route_difference_penalty = 0

    return route_difference_penalty


# Computation of profit - Objective function result
# function to compute shift excess penalty
def compute_shift_excess_penalty(routes_list, p_shift, distance_matrix, distance_route_vector, shift_duration):
    """
    Calculate penalty for routes exceeding the shift duration.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_shift (float): Penalty coefficient for overtime.
        distance_matrix (np.ndarray): Distance matrix.
        distance_route_vector (List[float]): Distances per route.
        shift_duration (float): Maximum allowed shift time (minutes).

    Returns:
        float: Total shift balance penalty.
    """
    route_time_vector = compute_route_time(routes_list, distance_route_vector)
    shift_excess = 0
    shift_excess_vector = []
    for i in route_time_vector:
        shift_excess = i - shift_duration
        if shift_excess <= 0:
            shift_excess_vector.append(0)
        else:
            shift_excess_vector.append(shift_excess)

        shift_excess = 0
    total_shift_excess = sum(shift_excess_vector)
    shift_excess_penalty = p_shift * total_shift_excess
    return shift_excess_penalty


# Computation of profit - Objective function result
# function to compute load excess penalty
def compute_load_excess_penalty(routes_list, p_load, data, vehicle_capacity):
    """
    Calculate penalty for vehicles exceeding their capacity limit.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_load (float): Penalty coefficient for excess load.
        data (pd.DataFrame): Bin weights data.
        vehicle_capacity (float): Max vehicle capacity.

    Returns:
        float: Total load penalty.
    """
    load_route = 0
    load_excess_route = 0
    total_load_excess = 0
    for i in routes_list:
        for j in i:
            stock = data["Stock"][j]
            accumulation_rate = data["Accum_Rate"][j]
            bin_stock = stock + accumulation_rate
            load_route += bin_stock
        load_excess_route = load_route - vehicle_capacity
        if load_excess_route > 0:
            total_load_excess += load_excess_route
        load_route = 0
        load_excess_route = 0
    load_excess_penalty = p_load * total_load_excess
    return load_excess_penalty


# Computation of profit - Objective function result
# function to compute profit
def compute_profit(
    routes_list,
    p_vehicle,
    p_load,
    p_route_difference,
    p_shift,
    data,
    distance_matrix,
    values,
):
    """
    Calculate the total objective value (Profit - Penalties).

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Vehicle penalty coefficient.
        p_load (float): Load penalty coefficient.
        p_route_difference (float): Imbalance penalty coefficient.
        p_shift (float): Shift penalty coefficient.
        data (pd.DataFrame): Bin data.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary containing keys like E, B, R, C,
            vehicle_capacity, and shift_duration.

    Returns:
        float: Total calculated profit.
    """
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values["E"], values["B"], values["R"])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values["C"])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list, p_vehicle)
    route_difference_penalty = compute_route_time_difference_penalty(
        routes_list, p_route_difference, distance_route_vector
    )
    load_excess_penalty = compute_load_excess_penalty(routes_list, p_load, data, values["vehicle_capacity"])
    shift_excess_penalty = compute_shift_excess_penalty(
        routes_list,
        p_shift,
        distance_matrix,
        distance_route_vector,
        values["shift_duration"],
    )

    profit = (
        total_revenue
        - transportation_cost
        - vehicle_penalty
        - route_difference_penalty
        - load_excess_penalty
        - shift_excess_penalty
    )
    return profit


# Profit without penalties
def compute_real_profit(routes_list, p_vehicle, data, distance_matrix, values):
    """
    Calculate the "real" profit (Revenue - Travel Cost - Vehicle penalty)
    excluding soft constraints.

    Args:
        routes_list (List[List[int]]): Current routing solution.
        p_vehicle (float): Vehicle penalty.
        data (pd.DataFrame): Bin data.
        distance_matrix (np.ndarray): Distance matrix.
        values (Dict): Parameter dictionary.

    Returns:
        float: Physical profit.
    """
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values["E"], values["B"], values["R"])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values["C"])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list, p_vehicle)

    profit = total_revenue - transportation_cost - vehicle_penalty
    return profit


# Lookahead sans policy functions
def compute_total_profit(routes, distance_matrix, id_to_index, data, R, V, density, cost_per_km=1.0):
    """
    Calcula o lucro total considerando:
    - Receita: soma dos pesos (kg) coletados multiplicado por R (€/kg)
    - Custo: soma das distâncias multiplicada por custo por km (default = 1.0 €/km)
    """
    total_kg = 0
    total_km = 0
    stocks = dict(zip(data["#bin"], data["Stock"]))
    for route in routes:
        if len(route) <= 2:
            continue
        # Calculate profit
        total_kg += (sum(stocks.get(b, 0) for b in route if b != 0) / MAX_CAPACITY_PERCENT) * V * (density)
        for i in range(len(route) - 1):
            a, b = route[i], route[i + 1]
            total_km += distance_matrix[id_to_index[a], id_to_index[b]]

    receita = total_kg * R
    custo = total_km * cost_per_km
    lucro = receita - custo
    return lucro, total_kg, total_km, receita


def compute_sans_route_cost(route, distance_matrix, id_to_index):
    """
    Calculate the travel cost/distance for a single route using id-to-index mapping.

    Args:
        route (List[int]): Sequence of node IDs.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict[int, int]): Mapping from node ID to matrix index.

    Returns:
        float: Route distance.
    """
    cost = 0
    for i in range(len(route) - 1):
        a, b = route[i], route[i + 1]
        cost += distance_matrix[id_to_index[a], id_to_index[b]]
    return cost


def compute_total_cost(routes, distance_matrix, id_to_index):
    """
    Calculate total travel distance for all routes.

    Args:
        routes (List[List[int]]): Set of routes.
        distance_matrix (np.ndarray): Distance matrix.
        id_to_index (Dict[int, int]): Node ID to matrix index mapping.

    Returns:
        float: Total distance.
    """
    return sum(compute_sans_route_cost(route, distance_matrix, id_to_index) for route in routes)
