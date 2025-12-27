# Functions for computations
# Computation of profit - Objective function result
# function to compute waste total revenue
def compute_waste_collection_revenue(routes_list, data, E, B, R):
    total_revenue_route = 0
    total_revenue = 0
    for i in routes_list:
        for j in i:
            stock = data['Stock'][j]
            accumulation_rate = data['Accum_Rate'][j]
            bin_stock = stock + accumulation_rate
            revenue_per_bin = bin_stock * E * B * R
            total_revenue_route += revenue_per_bin
        total_revenue += total_revenue_route
        total_revenue_route = 0
    return total_revenue


# Computation of profit - Objective function result
# function to compute distance travelled per route
def compute_distance_per_route(routes_list, distance_matrix):
    travelled_distance_route = 0
    distance_route_vector = []
    for i in routes_list:
        for j, element in enumerate(i):
            if j < len(i)-1:
                travelled_distance_route += distance_matrix[i[j]][i[j+1]]
        distance_route_vector.append(travelled_distance_route)
        travelled_distance_route = 0
    return distance_route_vector


# Computation of route time
# each bin in the route takes 3 minutes to be collected
# speed - 40km/h
def compute_route_time(routes_list, distance_route_vector):
    route_time_vector = []
    for route_n, i in enumerate(routes_list):
        route_time =  (len(i)-2)*3 + ((distance_route_vector[route_n]/40) * 60)
        route_time_vector.append(route_time)
    return route_time_vector


# Computation of profit - Objective function result
# function to compute transportation cost and distance per route
def compute_transportation_cost(routes_list, distance_route_vector, C):
    total_travelled_distance = sum(distance_route_vector)
    transportation_cost = C * total_travelled_distance
    return transportation_cost


# Computation of profit - Objective function result
# function to compute vehicle use penalty
# each route corresponds to one vehicle
def compute_vehicle_use_penalty(routes_list,p_vehicle):
    number_vehicles = len(routes_list)
    vehicle_penalty = p_vehicle * number_vehicles
    return vehicle_penalty


# Computation of profit - Objective function result
# function to compute difference between longest and shortest route length penalty
def compute_route_time_difference_penalty(routes_list,p_route_difference,distance_route_vector):
    route_time_vector = compute_route_time(routes_list,distance_route_vector)
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
    load_route = 0
    load_excess_route = 0
    total_load_excess = 0
    for i in routes_list:
        for j in i:
            stock = data['Stock'][j]
            accumulation_rate = data['Accum_Rate'][j]
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
def compute_profit(routes_list, p_vehicle, p_load, p_route_difference, p_shift, data, distance_matrix, values):
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values['E'], values['B'], values['R'])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values['C'])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list,p_vehicle)
    route_difference_penalty = compute_route_time_difference_penalty(routes_list,p_route_difference,distance_route_vector)
    load_excess_penalty = compute_load_excess_penalty(routes_list, p_load, data, values['vehicle_capacity'])
    shift_excess_penalty = compute_shift_excess_penalty(routes_list, p_shift, distance_matrix, distance_route_vector, values['shift_duration'])

    profit = total_revenue - transportation_cost - vehicle_penalty - route_difference_penalty - load_excess_penalty - shift_excess_penalty
    return profit


# Profit without penalties
def compute_real_profit(routes_list, p_vehicle, data, distance_matrix, values):
    distance_route_vector = compute_distance_per_route(routes_list, distance_matrix)
    total_revenue = compute_waste_collection_revenue(routes_list, data, values['E'], values['B'], values['R'])
    transportation_cost = compute_transportation_cost(routes_list, distance_route_vector, values['C'])
    vehicle_penalty = compute_vehicle_use_penalty(routes_list,p_vehicle)

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
    stocks = dict(zip(data['#bin'], data['Stock']))
    for route in routes:
        if len(route) <= 2:
            continue
        total_kg += ((sum(stocks.get(b, 0) for b in route if b != 0)/100) * V * (density))
        for i in range(len(route) - 1):
            a, b = route[i], route[i+1]
            total_km += distance_matrix[id_to_index[a], id_to_index[b]]
            
    receita = total_kg * R
    custo = total_km * cost_per_km
    lucro = receita - custo
    return lucro, total_kg, total_km, receita


def compute_sans_route_cost(route, distance_matrix, id_to_index):
    cost = 0
    for i in range(len(route) - 1):
        a, b = route[i], route[i+1]
        cost += distance_matrix[id_to_index[a], id_to_index[b]]
    return cost


def compute_total_cost(routes, distance_matrix, id_to_index):
    return sum(compute_sans_route_cost(route, distance_matrix, id_to_index) for route in routes)
