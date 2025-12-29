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


def local_search_2opt(tour, distance_matrix, max_iterations=200):
    """
    Standard 2-opt local search vectorized with NumPy.
    """
    if isinstance(tour, torch.Tensor):
        tour = tour.cpu().numpy()
    if torch.is_tensor(distance_matrix):
        distance_matrix = distance_matrix.cpu().numpy()
        
    best_tour = np.array(tour)
    n = len(best_tour)
    if n < 4:
        return best_tour.tolist()

    # Ensure it starts and ends at depot (0)
    if best_tour[0] != 0 or best_tour[-1] != 0:
        return best_tour.tolist()
    
    for _ in range(max_iterations):
        # i indices from 1 to n-3, j indices from i+1 to n-2
        i = np.arange(1, n - 2)
        j = np.arange(2, n - 1)
        
        I, J = np.meshgrid(i, j, indexing='ij')
        mask = J > I
        
        if not np.any(mask):
            break
            
        I_vals = I[mask]
        J_vals = J[mask]
        
        # Tour nodes at relevant indices
        t_prev_i = best_tour[I_vals - 1]
        t_curr_i = best_tour[I_vals]
        t_curr_j = best_tour[J_vals]
        t_next_j = best_tour[J_vals + 1]
        
        # Gain calculation: current_dist - new_dist
        d_curr = distance_matrix[t_prev_i, t_curr_i] + distance_matrix[t_curr_j, t_next_j]
        d_next = distance_matrix[t_prev_i, t_curr_j] + distance_matrix[t_curr_i, t_next_j]
        gains = d_curr - d_next
        
        best_idx = np.argmax(gains)
        best_gain = gains[best_idx]
        if best_gain > 1e-5:
            # Apply the best edge swap found in this iteration
            target_i = I_vals[best_idx]
            target_j = J_vals[best_idx]
            best_tour[target_i : target_j + 1] = best_tour[target_i : target_j + 1][::-1]
        else:
            break
            
    return best_tour.tolist()


def local_search_2opt_vectorized(tours, distance_matrix, max_iterations=200):
    """
    Vectorized 2-opt local search across a batch of tours using PyTorch.
    Optimized to perform edge swaps for all batch instances in parallel.
    """
    device = distance_matrix.device
    
    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)
    
    # Handle distance_matrix expansion
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0)
    
    B, N = tours.shape
    if N < 4:
        return tours if is_batch else tours.squeeze(0)

    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)
        
    batch_indices = torch.arange(B, device=device).view(B, 1)
    
    for _ in range(max_iterations):
        # Generate indices for all possible edge swaps (i, j)
        indices = torch.arange(N, device=device)
        i = indices[1:-2]
        j = indices[2:-1]
        
        I, J = torch.meshgrid(i, j, indexing='ij')
        mask = J > I
        if not mask.any():
            break
            
        I_vals = I[mask]
        J_vals = J[mask]
        K = I_vals.size(0)
        
        # Tour nodes at relevant indices: (B, K)
        t_prev_i = tours[:, I_vals - 1]
        t_curr_i = tours[:, I_vals]
        t_curr_j = tours[:, J_vals]
        t_next_j = tours[:, J_vals + 1]
        
        # Gain calculation: (B, K)
        # Use advanced indexing for batch
        b_idx_exp = batch_indices.expand(B, K)
        d_curr = distance_matrix[b_idx_exp, t_prev_i, t_curr_i] + distance_matrix[b_idx_exp, t_curr_j, t_next_j]
        d_next = distance_matrix[b_idx_exp, t_prev_i, t_curr_j] + distance_matrix[b_idx_exp, t_curr_i, t_next_j]
        gains = d_curr - d_next
        
        # Find best gain for each instance in the batch
        best_gain, best_idx = torch.max(gains, dim=1)
        
        # Determine which instances actually improved
        improved = best_gain > 1e-5
        if not improved.any():
            break
            
        # Parallel segment reversal
        # Construct transform indices (B, N)
        target_i = I_vals[best_idx]
        target_j = J_vals[best_idx]
        
        k = torch.arange(N, device=device).view(1, N).expand(B, N)
        idx_map = torch.arange(N, device=device).view(1, N).expand(B, N).clone()
        
        # For instances that improved, reverse the [target_i, target_j] range
        # reversal_mask: (B, N)
        reversal_range_mask = (k >= target_i.view(B, 1)) & (k <= target_j.view(B, 1))
        reversal_mask = reversal_range_mask & improved.view(B, 1)
        
        # idx[b, k] = target_i[b] + target_j[b] - k
        rev_idx = target_i.view(B, 1) + target_j.view(B, 1) - k
        idx_map[reversal_mask] = rev_idx[reversal_mask]
        
        # Apply the best edge swap for all batch elements simultaneously
        tours = torch.gather(tours, 1, idx_map)
        
    return tours if is_batch else tours.squeeze(0)


def get_route_cost(distancesC, tour):
    if isinstance(tour, torch.Tensor) and isinstance(distancesC, torch.Tensor):
        return distancesC[tour[:-1], tour[1:]].sum().cpu().numpy().item()
    else:
        distancesC2 = distancesC.copy() if isinstance(distancesC, np.ndarray) else np.array(distancesC)
        tour2 = tour.copy() if isinstance(tour, np.ndarray) else np.array(tour)
        return np.sum(distancesC2[tour2[:-1], tour2[1:]]).item()


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
