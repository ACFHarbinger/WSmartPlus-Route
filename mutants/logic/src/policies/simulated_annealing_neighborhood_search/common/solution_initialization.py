"""
Initialization strategies for vehicle routing solutions.
Extracts constructive heuristics from the legacy solutions module.
"""

import numpy as np


def find_initial_solution(data, bins_coordinates, distance_matrix, number_of_bins, vehicle_capacity, E, B):
    """
    Construct a feasible initial solution for the routing problem.

    Processes overflow predictions and builds routes using a construction heuristic.

    Args:
        data (pd.DataFrame): Bin weights and metadata.
        bins_coordinates (List): Lat/Lon pairs.
        distance_matrix (np.ndarray): Shortest path distances.
        number_of_bins (int): Scale of the problem.
        vehicle_capacity (float): Max tanker capacity.
        E (float): Bin volume.
        B (float): Bin density.

    Returns:
        Tuple: (routes, removed_bins, bins_cannot_removed, points).
    """
    bins = list(data["#bin"][1 : number_of_bins + 1])
    depot = list(data["#bin"][0:1])

    # Initialization of routes
    i = 0
    routes_list = list()
    max_lng = max(bins_coordinates["Lng"])
    min_lng = min(bins_coordinates["Lng"])
    lng_amp = max_lng - min_lng
    zone_1_s = min_lng
    zone_1_l = min_lng + lng_amp / 3
    zone_2_l = min_lng + 2 * lng_amp / 3
    lng_list = list(bins_coordinates["Lng"])
    lng_list_without_dep = lng_list[1 : len(lng_list)]
    bins_zone_1 = []
    bins_zone_2 = []
    bins_zone_3 = []
    for n, h in enumerate(lng_list_without_dep):
        if h >= zone_1_s and h < zone_1_l:
            bins_zone_1.append(n + 1)
        elif h >= zone_1_l and h < zone_2_l:
            bins_zone_2.append(n + 1)
        else:
            bins_zone_3.append(n + 1)
    while len(bins) != 0:
        i += 1
        globals()["route_{0}".format(i)] = []
        space_occupied = 0

        # Choose depot to initialize the route
        bin_chosen_n = depot[0]

        # Get data for the bin chosen
        corresponding_row = data[data["#bin"] == bin_chosen_n]

        # Get fill level of the bin (stock)
        stock = (corresponding_row.iloc[0]["Stock"] + corresponding_row.iloc[0]["Accum_Rate"]) * E * B
        space_occupied += stock
        globals()["route_{0}".format(i)].append(bin_chosen_n)

        # Previous_bin is the bin previously added in the route
        previous_bin = bin_chosen_n

        # While there is space in the vehicle and there are bins to collect:
        while space_occupied < vehicle_capacity and len(bins) != 0:
            old_space_occupied = space_occupied
            while space_occupied < vehicle_capacity and len(bins_zone_1) != 0:
                # Get the distance between the previous bin and all the other bins
                row = distance_matrix[previous_bin][:]

                # Delete previous bin to previous bin distance
                row_new = np.delete(row, previous_bin)

                # Get the index of the bin with the minimum distance from the previous one
                min(row_new)
                min_idx = row_new.argmin()

                # Get the correct bin id (because of deleting the 0 distance)
                if min_idx >= previous_bin:
                    next_bin_idx = min_idx + 1
                else:
                    next_bin_idx = min_idx

                # Check if the closest bin is already in any of the routes created
                stop = None
                if (
                    next_bin_idx in globals()["route_{0}".format(i)]
                    or next_bin_idx not in bins
                    or next_bin_idx not in bins_zone_1
                ):
                    # if yes, sort the distances from the shortest to the longest
                    row_sorted = np.sort(row_new)
                    j = 1
                    stop = "A"

                    # Iterate through sorted list until the closest bin that is not in any route is found
                    while j <= len(row_sorted[1:]) and stop != "B":
                        row_new_list = row_new.tolist()

                        # Get index of the bin that is being tried
                        idx_tried_bin = row_new_list.index(row_sorted[j])
                        if idx_tried_bin >= previous_bin:
                            next_bin_idx = idx_tried_bin + 1
                        else:
                            next_bin_idx = idx_tried_bin

                        # Verify if it is in any route and not only in the current route
                        if (
                            next_bin_idx not in globals()["route_{0}".format(i)]
                            and next_bin_idx in bins
                            and next_bin_idx in bins_zone_1
                        ):
                            stop = "B"
                            min_idx = next_bin_idx

                        j += 1

                if stop == "A" and (
                    next_bin_idx in globals()["route_{0}".format(i)] or next_bin_idx not in bins_zone_1
                ):
                    break

                # Get current bin index from the bins list
                try:
                    bin_index_in_bins = bins.index(next_bin_idx)
                except ValueError:
                    break

                # Update space occupied in the vehicle
                corresponding_row = data[data["#bin"] == next_bin_idx]
                stock = (corresponding_row.iloc[0]["Stock"] + corresponding_row.iloc[0]["Accum_Rate"]) * E * B
                space_occupied += stock
                if space_occupied < vehicle_capacity and next_bin_idx not in globals()["route_{0}".format(i)]:
                    # Add bin to the route
                    globals()["route_{0}".format(i)].append(next_bin_idx)
                    bins.pop(bin_index_in_bins)

                    # Remove bin from bins zone
                    bin_index_in_bins_zone = bins_zone_1.index(next_bin_idx)
                    bins_zone_1.pop(bin_index_in_bins_zone)

                    # Update previous bin
                    previous_bin = next_bin_idx
            while space_occupied < vehicle_capacity and len(bins_zone_2) != 0:
                # Get the distance between the previous bin and all the other bins
                row = distance_matrix[previous_bin][:]

                # Delete previous bin to previous bin distance
                row_new = np.delete(row, previous_bin)

                # Get the index of the bin with the minimum distance from the previous one
                min(row_new)
                min_idx = row_new.argmin()

                # Get the correct bin id (because of deleting the 0 distance)
                if min_idx >= previous_bin:
                    next_bin_idx = min_idx + 1
                else:
                    next_bin_idx = min_idx

                # Check if the closest bin is already in any of the routes created
                stop = None
                if (
                    next_bin_idx in globals()["route_{0}".format(i)]
                    or next_bin_idx not in bins
                    or next_bin_idx not in bins_zone_2
                ):
                    # if yes, sort the distances from the shortest to the longest
                    row_sorted = np.sort(row_new)
                    j = 1
                    stop = "A"

                    # Iterate through sorted list until the closest bin that is not in any route is found
                    while j <= len(row_sorted[1:]) and stop != "B":
                        row_new_list = row_new.tolist()

                        # Get index of the bin that is being tried
                        idx_tried_bin = row_new_list.index(row_sorted[j])
                        if idx_tried_bin >= previous_bin:
                            next_bin_idx = idx_tried_bin + 1
                        else:
                            next_bin_idx = idx_tried_bin

                        # Verify if it is in any route and not only in the current route
                        if (
                            next_bin_idx not in globals()["route_{0}".format(i)]
                            and next_bin_idx in bins
                            and next_bin_idx in bins_zone_2
                        ):
                            stop = "B"
                            min_idx = next_bin_idx

                        j += 1

                if stop == "A" and (
                    next_bin_idx in globals()["route_{0}".format(i)] or next_bin_idx not in bins_zone_2
                ):
                    break

                # Get current bin index from the bins list
                try:
                    bin_index_in_bins = bins.index(next_bin_idx)
                except ValueError:
                    break

                # Update space occupied in the vehicle
                corresponding_row = data[data["#bin"] == next_bin_idx]
                stock = (corresponding_row.iloc[0]["Stock"] + corresponding_row.iloc[0]["Accum_Rate"]) * E * B
                space_occupied += stock
                if space_occupied < vehicle_capacity and next_bin_idx not in globals()["route_{0}".format(i)]:
                    # Add bin to the route
                    globals()["route_{0}".format(i)].append(next_bin_idx)

                    # Remove bin from bins to be collected list
                    bins.pop(bin_index_in_bins)

                    # Remove bin from bins zone
                    bin_index_in_bins_zone = bins_zone_2.index(next_bin_idx)
                    bins_zone_2.pop(bin_index_in_bins_zone)

                    # Update previous bin
                    previous_bin = next_bin_idx

            while space_occupied < vehicle_capacity and len(bins_zone_3) != 0:
                # Get the distance between the previous bin and all the other bins
                row = distance_matrix[previous_bin][:]
                row_sorted = np.sort(row)
                j = 1
                stop = "A"

                # Iterate through sorted list until the closest bin that is not in any route is found
                while j <= len(row_sorted[1:]) and stop != "B":
                    _ = row_sorted.tolist()
                    possible_indexes = [index for index, value in enumerate(row) if value == row_sorted[j]]
                    for z in possible_indexes:
                        if z in bins:
                            idx_tried_bin = z
                            next_bin_idx = z

                    # Verify if it is in any route and not only in the current route
                    if (
                        next_bin_idx not in globals()["route_{0}".format(i)]
                        and next_bin_idx in bins
                        and next_bin_idx in bins_zone_3
                    ):
                        stop = "B"
                        min_idx = next_bin_idx

                    j += 1

                if stop == "A" and (
                    next_bin_idx in globals()["route_{0}".format(i)] or next_bin_idx not in bins_zone_3
                ):
                    break

                # Get current bin index from the bins list
                try:
                    bin_index_in_bins = bins.index(next_bin_idx)
                except ValueError:
                    break

                # Update space occupied in the vehicle
                corresponding_row = data[data["#bin"] == next_bin_idx]
                stock = (corresponding_row.iloc[0]["Stock"] + corresponding_row.iloc[0]["Accum_Rate"]) * E * B
                space_occupied += stock
                if space_occupied < vehicle_capacity and next_bin_idx not in globals()["route_{0}".format(i)]:
                    # Add bin to the route
                    globals()["route_{0}".format(i)].append(next_bin_idx)

                    # Remove bin from bins to be collected list
                    bins.pop(bin_index_in_bins)

                    # Remove bin from bins zone
                    bin_index_in_bins_zone = bins_zone_3.index(next_bin_idx)
                    bins_zone_3.pop(bin_index_in_bins_zone)

                    # Update previous bin
                    previous_bin = next_bin_idx
                # else:
                # travel_time = travel_time - (distance/37.5) * 60 + 5

            if space_occupied == old_space_occupied:
                break

        globals()["route_{0}".format(i)] = globals()["route_{0}".format(i)] + depot
        routes_list.append(globals()["route_{0}".format(i)])
    return routes_list


def compute_initial_solution(data, bins_coordinates, distance_matrix, vehicle_capacity, id_to_index):
    """
    Simplified initial solution builder for SANS policy.

    Args:
        data (pd.DataFrame): Bin metadata.
        bins_coordinates (List): Coordinates.
        distance_matrix (np.ndarray): Distances.
        vehicle_capacity (float): Tanker capacity.
        id_to_index (Dict): Mapping.

    Returns:
        Tuple: (initial_routes, removed_bins).
    """
    depot = data["#bin"].iloc[0]
    all_bins = list(data["#bin"][1:])
    bins_coordinates = {b: coord for b, coord in bins_coordinates.items() if b in all_bins or b == depot}
    stocks = dict(zip(data["#bin"], data["Stock"]))

    rotas = []
    bins_restantes = set(all_bins)

    # Enquanto houver bins a serem coletados
    while bins_restantes:
        # Inicializa a rota a partir do depósito
        current_route = [depot]
        carga = 0
        atual = depot
        bins_restantes_inicial = set(bins_restantes)

        # Enquanto houver bins restantes e espaço no veículo
        while bins_restantes_inicial:
            # Filtra os bins que podem ser adicionados sem ultrapassar a capacidade do veículo
            candidatos = [b for b in bins_restantes_inicial if carga + stocks[b] <= vehicle_capacity]
            if not candidatos:
                break

            # Encontra o bin mais próximo
            idx_atual = id_to_index[atual]
            proximo = min(candidatos, key=lambda b: distance_matrix[idx_atual][id_to_index[b]])

            # Adiciona o bin mais próximo à rota
            current_route.append(proximo)
            carga += stocks[proximo]
            bins_restantes_inicial.remove(proximo)
            atual = proximo

        # Retorna ao depósito
        current_route.append(depot)

        # Se a rota tiver mais de 2 pontos (depot + bins), adiciona à lista de rotas
        if len(current_route) > 2:
            rotas.append(current_route)

        # Remove os bins que já foram coletados
        bins_restantes = bins_restantes - set(current_route[1:-1])
    return rotas
