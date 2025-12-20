import torch
import fast_tsp
import numpy as np
import networkx as nx


def find_route(C, to_collect):
    to_collect_tmp = [0] + list(to_collect)
    tmpC = C[to_collect_tmp, :][:, to_collect_tmp]
    tour = fast_tsp.find_tour(tmpC)
    zero_index = tour.index(0)
    tour = tour[zero_index:] + tour[:zero_index]
    #cost = fast_tsp.compute_cost(tour, tmpC)
    tour2 = []
    for ii in range(0, len(tour) - 1):
        current_node = to_collect_tmp[tour[ii]]
        next_node = to_collect_tmp[tour[ii + 1]]
        tour2.append(current_node)
    tour2.extend([next_node, 0])
    return tour2


def get_route_cost(distancesC, tour):
    if isinstance(tour, torch.Tensor) and isinstance(distancesC, torch.Tensor):
        return distancesC[tour[:-1], tour[1:]].sum().cpu().numpy()
    else:
        distancesC2 = distancesC.copy() if isinstance(distancesC, np.ndarray) else np.array(distancesC)
        tour2 = tour.copy() if isinstance(tour, np.ndarray) else np.array(tour)
        return np.sum(distancesC2[tour2[:-1], tour2[1:]])


def get_path_cost(G, p):
    l = p[0]
    c = 0
    for id_i in range(1, len(p)):
        try:
            c += G.get_edge_data(l, p[id_i])['weight']
        except:
            c += 1
        l = p[id_i]
    return c


def get_multi_tour(tour, bins_waste, max_capacity, distance_matrix):
    depot_trips = 0
    final_tour = tour
    vehicle_collected = 0
    tmp_tour = [x - 1 for x in tour if x != 0]
    for i in range(len(tmp_tour)):
        cur_bin = tmp_tour[i]
        col_waste = bins_waste[cur_bin]
        if vehicle_collected + col_waste < max_capacity:
            vehicle_collected += col_waste
        elif vehicle_collected + col_waste > max_capacity:
            final_tour.insert(i + depot_trips, 0)
            vehicle_collected = col_waste
            depot_trips += 1
            #cost += distance_matrix[tmp_tour[i - 1], 0] + distance_matrix[0, cur_bin]
        else:
            final_tour.insert(i + depot_trips - 1, 0)
            vehicle_collected = 0
            depot_trips += 1
            #if i < len(tmp_tour) - 1: 
                #cost += distance_matrix[cur_bin, 0] + distance_matrix[0, tmp_tour[i + 1]]
    return final_tour


def get_partial_tour(tour, bins, max_capacity, distance_matrix, cost):
    tmp_tour = [x - 1 for x in tour if x != 0]
    total_waste = np.sum(bins[tmp_tour])
    while total_waste > max_capacity:
        min_waste_bin_idx = np.argmin(bins[tmp_tour])
        bin_to_remove = tmp_tour[min_waste_bin_idx]
        total_waste -= bins[bin_to_remove]
        cost -= distance_matrix[tmp_tour[min_waste_bin_idx - 1], bin_to_remove]
        tmp_tour = np.delete(tmp_tour, min_waste_bin_idx)
    return tmp_tour, cost


# Create matrix will all distances
def dist_matrix_from_graph(G):
    paths_between_states = []
    n_vertices = len(G.nodes)
    dist_matrix = np.zeros((n_vertices, n_vertices), int)
    for id_i in range(n_vertices):
        paths_between_states.append([])
        for id_j in range(n_vertices):
            if id_i == id_j:
                paths_between_states[id_i].append([])
                continue
            p = nx.dijkstra_path(G, source = id_i, target = id_j)
            paths_between_states[id_i].append(p)
            dist_matrix[id_i, id_j] = int(get_path_cost(G, p))
    return dist_matrix, paths_between_states
