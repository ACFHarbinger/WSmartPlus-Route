"""
Vectorized Split Algorithms for Vehicle Routing Problems.

This module implements the Split algorithm, which optimally partitions a 'giant tour'
(a permutation of all nodes) into a sequence of feasible routes that satisfy
vehicle capacity constraints.

Key Implementations:
- vectorized_linear_split: Optimal splitting using Bellman-Ford on a DAG.
- _vectorized_split_limited: Split algorithm restricted to a maximum number of vehicles.
- _reconstruct_routes: Reconstructs route lists from the split predecessors.
"""

import torch


def vectorized_linear_split(giant_tours, dist_matrix, demands, vehicle_capacity, max_len=None, max_vehicles=None):
    """
    Vectorized Linear Split Algorithm for Batch of Giant Tours.
    Computes the optimal segmentation of giant tours into routes subject to capacity constraints.

    Args:
        giant_tours: (B, N) tensor of node indices (permutation of 1..N or subset).
        dist_matrix: (B, N_all, N_all) tensor of distances.
        demands: (B, N_all) tensor of demands.
        vehicle_capacity: float, scalar or (B,) tensor.
        max_len: int, maximum route length to consider for efficiency (default: N).

    Returns:
        routes: (B, list_of_routes) where each route is a list/tensor of nodes with depot 0.
        costs: (B,) total cost of the split.
    """
    B, N = giant_tours.size()
    device = giant_tours.device

    if max_len is None:
        max_len = N

    # Standardize inputs
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    if demands.dim() == 1:
        demands = demands.unsqueeze(0).expand(B, -1)

    if isinstance(vehicle_capacity, torch.Tensor) and vehicle_capacity.dim() == 1:
        vehicle_capacity = vehicle_capacity.unsqueeze(1)

    # Precompute accumulations
    tour_demands = torch.gather(demands, 1, giant_tours)
    cum_load = torch.cumsum(tour_demands, dim=1)

    from_nodes = giant_tours[:, :-1]
    to_nodes = giant_tours[:, 1:]
    batch_ids = torch.arange(B, device=device).view(B, 1)
    tour_dists = dist_matrix[batch_ids, from_nodes, to_nodes]
    tour_dists = torch.cat([torch.zeros((B, 1), device=device), tour_dists], dim=1)
    cum_dist = torch.cumsum(tour_dists, dim=1)

    d_0_i = dist_matrix[batch_ids, 0, giant_tours]
    d_i_0 = dist_matrix[batch_ids, giant_tours, 0]

    cum_load_pad = torch.cat([torch.zeros((B, 1), device=device), cum_load], dim=1)
    cum_dist_pad = torch.cat([torch.zeros((B, 1), device=device), cum_dist], dim=1)

    # Check for max_vehicles constraint
    if max_vehicles and max_vehicles > 0:
        return _vectorized_split_limited(
            B,
            N,
            device,
            max_vehicles,
            vehicle_capacity,
            cum_load_pad,
            cum_dist_pad,
            d_0_i,
            d_i_0,
            giant_tours,
        )

    # 2. Bellman-Ford / Shortest Path on DAG (Unlimited Vehicles)
    V = torch.full((B, N + 1), float("inf"), device=device)
    V[:, 0] = 0
    P = torch.full((B, N + 1), -1, dtype=torch.long, device=device)

    for i in range(1, N + 1):
        j_start = max(0, i - max_len)
        j_end = i

        js = torch.arange(j_start, j_end, device=device).view(1, -1).expand(B, -1)

        loads = cum_load_pad[:, i : i + 1] - torch.gather(cum_load_pad, 1, js)
        mask = loads <= vehicle_capacity

        dist_0_first = torch.gather(d_0_i, 1, js)
        dist_last_0 = d_i_0[:, i - 1 : i]
        path_dists = cum_dist_pad[:, i : i + 1] - torch.gather(cum_dist_pad, 1, js + 1)
        segment_costs = dist_0_first + path_dists + dist_last_0

        total_costs = torch.gather(V, 1, js) + segment_costs
        total_costs[~mask] = float("inf")

        min_vals, min_idxs = torch.min(total_costs, dim=1)
        V[:, i] = min_vals

        best_js = torch.gather(js, 1, min_idxs.unsqueeze(1)).squeeze(1)
        P[:, i] = best_js

    # 3. Path Reconstruction
    return _reconstruct_routes(B, N, giant_tours, P, V[:, N])


def _vectorized_split_limited(
    B,
    N,
    device,
    max_vehicles,
    capacity,
    cum_load,
    cum_dist_pad,
    d_0_i,
    d_i_0,
    giant_tours,
):
    """
    Limited split using K-step updates (like K shortest path layers).
    Constrains the solution to use at most `max_vehicles` routes.

    Args:
        B (int): Batch size.
        N (int): Number of nodes.
        device: Torch device.
        max_vehicles (int): Maximum number of vehicles allowed.
        capacity (float): Vehicle capacity.
        cum_load (torch.Tensor): Cumulative load tensor.
        cum_dist_pad (torch.Tensor): Cumulative distance tensor (padded).
        d_0_i (torch.Tensor): Distance from depot to nodes.
        d_i_0 (torch.Tensor): Distance from nodes to depot.
        giant_tours (torch.Tensor): Giant tour indices.

    Returns:
        tuple: (routes_batch, costs)
    """
    # Precompute Cost Matrix for all i, j?
    # Or batched operations?
    # Full Cost Matrix (B, N+1, N+1) is better for vectorization if N is small (<500).

    # Create grid of (j, i)
    # j < i.
    # We want Cost[j, i] = cost of segment from j to i-1 (indices in giant tour).

    # Expand dims
    # j: source (0..N-1), i: dest (1..N)
    # We construct full matrix and mask.

    # i range: 1..N. j range: 0..N-1.
    # Cost Matrix will be (B, N_start, N_end) where starts are 0..N-1, ends are 1..N?
    # Easier: (B, N+1, N+1).

    indices = torch.arange(N + 1, device=device)
    J = indices.view(1, N + 1, 1).expand(B, N + 1, N + 1)
    idx_i = indices.view(1, 1, N + 1).expand(B, N + 1, N + 1)

    # Mask valid upper triangle: j < i
    mask_indices = J < idx_i

    # Load Constraint
    # Load(j, i) = cum_load[i] - cum_load[j]
    # We want Load[j, i] -> cum_load[i] - cum_load[j]
    # cum_load shape (B, N+1).
    # loads[b, j, i] = cum_load[b, i] - cum_load[b, j] ?
    # Let's align dimensions.
    # Expanding cum_load to (B, 1, N+1) gives dim 2 varying i.
    # Expanding cum_load to (B, N+1, 1) gives dim 1 varying j.
    # L[b, j, i] = cum_load[b, i] - cum_load[b, j].
    # Yes.

    L = cum_load.unsqueeze(1) - cum_load.unsqueeze(2)
    # Transposing to align with [B, j, i]?
    # result of unsqueeze(1) is (B, 1, N+1) -> values along i.
    # result of unsqueeze(2) is (B, N+1, 1) -> values along j.
    # (B, 1, N+1) - (B, N+1, 1) -> (B, N+1, N+1).
    # Element [b, j, i] = load[b, i] - load[b, j]. Correct.

    mask_cap = (L <= capacity) & mask_indices

    # Costs
    # SegCost[j, i] = d(0, T[j]) + Path(j, i) + d(T[i-1], 0)
    # T[j] is first node. T[i-1] is last node.
    # d_0_i is size (B, N). Corresponds to T[0]..T[N-1].
    # We need to map j to d_0_i index.
    # Index in d_0_i is j.
    # Index in d_i_0 is i-1.

    # We need to handle j=N? j ranges 0..N-1 effectively.
    # If j=N, it's invalid source for loop (must end at N).

    # Prepare d0, dlast
    # Pad d_0_i, d_i_0 to N+1 for safe indexing?
    # d_0_Tj (B, j).
    # d_Ti_1_0 (B, i).

    d0_pad = torch.cat([d_0_i, torch.zeros((B, 1), device=device)], dim=1)  # (B, N+1)
    # Careful with indices.
    # d(0, T[j]): indices 0..N-1.
    # d(T[i-1], 0): indices 0..N-1.
    # i goes 1..N. i-1 goes 0..N-1.
    # j goes 0..N-1.

    # Matrix Broadcast
    # D_Start[j, i] depends only on j.
    D_Start = d0_pad.unsqueeze(2).expand(-1, -1, N + 1)  # (B, N+1, N+1) along j

    # D_End[j, i] depends only on i.
    # We need index i-1.
    # Shift d_i_0 right?
    # d_end_vec[i] = d_i_0[i-1].
    # d_end_vec[0] dummy.
    d_end_vec = torch.cat([torch.zeros((B, 1), device=device), d_i_0], dim=1)
    D_End = d_end_vec.unsqueeze(1).expand(-1, N + 1, -1)  # (B, N+1, N+1) along i

    # Path Dist[j, i] = cum_dist[i] - cum_dist[j+1]
    # cum_dist indices: i maps to end, j+1 maps to start?
    # Verified in main func: path_dists = cum_dist_pad[i] - cum_dist_pad[j+1]
    # P[b, j, i] = cum_dist_pad[b, i] - cum_dist_pad[b, j+1]

    # We need cum_dist_pad shifted?
    # cum_dist_pad has N+1.
    # j+1 goes 1..N+1? if j=N, j+1=N+1 (out of bounds).
    # Pad extra?
    cd_pad = torch.cat([cum_dist_pad, torch.zeros((B, 1), device=device)], dim=1)  # (B, N+2)

    # CD_I[b, j, i] = cum_dist_pad[b, i]
    # CD_J[b, j, i] = cd_pad[b, j+1]

    CD_I = cum_dist_pad.unsqueeze(1).expand(-1, N + 1, -1)

    # For J+1, we shift/roll dim 1 of cum_dist_pad? Or just index?
    # Unfold?
    # Just construct vector: [cd[1], cd[2]... cd[N], 0].
    cd_j_vec = cd_pad[:, 1 : N + 2]  # (B, N+1)
    CD_J = cd_j_vec.unsqueeze(2).expand(-1, -1, N + 1)

    PathCosts = CD_I - CD_J

    Costs = D_Start + PathCosts + D_End
    Costs[~mask_cap] = float("inf")

    # Optimization Loop
    # V[k, i] = min cost to reach i using k vehicles.
    # Init
    V = torch.full((B, N + 1), float("inf"), device=device)
    V[:, 0] = 0

    # Storage for reconstruction
    # P[k, i] = predecessor j
    P_k = torch.full((B, max_vehicles + 1, N + 1), -1, dtype=torch.long, device=device)

    best_cost = torch.full((B,), float("inf"), device=device)
    best_k_idx = torch.full((B,), -1, dtype=torch.long, device=device)

    # This might be memory hungry (B, N, N) for cost matrix.
    # With N=100, B=128 => 128*100*100 * 4 = 5MB. OK.

    for k in range(1, max_vehicles + 1):
        # V_new[i] = min_j (V_old[j] + Costs[j, i])
        # Broadcast V_old (B, j) -> (B, j, 1) -> (B, j, N+1)
        # Add to Costs (B, j, i)
        # Min over dim 1 (j)

        M = V.unsqueeze(2) + Costs
        min_vals, min_idxs = M.min(dim=1)  # (B, i)

        V_next = min_vals

        # Store
        P_k[:, k, :] = min_idxs
        V = V_next

        # Check finish?
        # If we want *exactly* k? Or <= k?
        # If <= k, we just take best of all k?
        # Typically HGS minimizes cost. Less vehicles usually better if cost lower.
        # But we track best feasible solution.

        # If V[:, N] is better than best_cost, update
        improved = V[:, N] < best_cost
        best_cost[improved] = V[improved, N]
        best_k_idx[improved] = k

    return _reconstruct_limited(B, N, giant_tours, P_k, best_k_idx, best_cost)


def _reconstruct_routes(B, N, giant_tours, P, costs):
    """
    Reconstructs the routes from the predecessor matrix P obtained from the Split algorithm.

    Args:
        B (int): Batch size.
        N (int): Number of nodes.
        giant_tours (torch.Tensor): Giant tour indices.
        P (torch.Tensor): Predecessor matrix (B, N+1).
        costs (torch.Tensor): Costs vector (B,).

    Returns:
        tuple: (list of routes per batch item, costs)
    """
    routes_batch = []
    P_cpu = P.cpu().numpy()
    giant_tours_cpu = giant_tours.cpu().numpy()

    for b in range(B):
        # ... (Same logic as original, just moved here) ...
        curr = N
        route_nodes = []
        possible = True
        while curr > 0:
            prev = P_cpu[b, curr]
            if prev == -1:
                possible = False
                break
            nodes = giant_tours_cpu[b, prev:curr]
            route_nodes = [0] + list(nodes) + route_nodes
            curr = prev

        if route_nodes and route_nodes[-1] != 0:
            route_nodes.append(0)

        if not possible:
            # Fallback or empty?
            pass

        routes_batch.append(route_nodes)
    return routes_batch, costs


def _reconstruct_limited(B, N, giant_tours, P_k, best_k, costs):
    """
    Reconstructs routes for the limited vehicle split algorithm.

    Args:
        B (int): Batch size.
        N (int): Number of nodes.
        giant_tours (torch.Tensor): Giant tour indices.
        P_k (torch.Tensor): Predecessor matrix with k dimension (B, max_k + 1, N + 1).
        best_k (torch.Tensor): Index of the best number of vehicles used for each batch item.
        costs (torch.Tensor): Costs vector.

    Returns:
        tuple: (list of routes, costs)
    """
    routes_batch = []
    P_cpu = P_k.cpu().numpy()
    k_cpu = best_k.cpu().numpy()
    giant_tours_cpu = giant_tours.cpu().numpy()

    for b in range(B):
        k = k_cpu[b]
        if k == -1 or costs[b] == float("inf"):
            routes_batch.append([])
            continue

        curr = N
        route_nodes = []

        # Backtrack with known k
        current_k = k

        while curr > 0 and current_k > 0:
            prev = P_cpu[b, current_k, curr]
            if prev == -1:
                break

            nodes = giant_tours_cpu[b, prev:curr]
            route_nodes = [0] + list(nodes) + route_nodes

            curr = prev
            current_k -= 1

        if route_nodes and route_nodes[-1] != 0:
            route_nodes.append(0)

        routes_batch.append(route_nodes)

    return routes_batch, costs
