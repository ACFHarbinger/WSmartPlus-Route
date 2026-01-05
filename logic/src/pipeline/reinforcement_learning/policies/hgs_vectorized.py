import torch
import time


from logic.src.pipeline.reinforcement_learning.core.post_processing import local_search_2opt_vectorized

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
        return _vectorized_split_limited(B, N, device, max_vehicles, vehicle_capacity, 
                                         cum_load_pad, cum_dist_pad, d_0_i, d_i_0, giant_tours)
    
    # 2. Bellman-Ford / Shortest Path on DAG (Unlimited Vehicles)
    V = torch.full((B, N + 1), float('inf'), device=device)
    V[:, 0] = 0
    P = torch.full((B, N + 1), -1, dtype=torch.long, device=device)
    
    for i in range(1, N + 1):
        j_start = max(0, i - max_len)
        j_end = i
        
        js = torch.arange(j_start, j_end, device=device).view(1, -1).expand(B, -1)
        
        loads = cum_load_pad[:, i:i+1] - torch.gather(cum_load_pad, 1, js)
        mask = loads <= vehicle_capacity
        
        dist_0_first = torch.gather(d_0_i, 1, js) 
        dist_last_0 = d_i_0[:, i-1:i]
        path_dists = cum_dist_pad[:, i:i+1] - torch.gather(cum_dist_pad, 1, js + 1)
        segment_costs = dist_0_first + path_dists + dist_last_0
        
        total_costs = torch.gather(V, 1, js) + segment_costs
        total_costs[~mask] = float('inf')
        
        min_vals, min_idxs = torch.min(total_costs, dim=1)
        V[:, i] = min_vals
        
        best_js = torch.gather(js, 1, min_idxs.unsqueeze(1)).squeeze(1)
        P[:, i] = best_js

    # 3. Path Reconstruction
    return _reconstruct_routes(B, N, giant_tours, P, V[:, N])


def _vectorized_split_limited(B, N, device, max_vehicles, capacity, cum_load, cum_dist_pad, d_0_i, d_i_0, giant_tours):
    """
    Limited split using K-step updates (like K shortest path layers).
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
    i_idx = torch.arange(1, N + 1, device=device).view(1, 1, N).expand(B, N, N) # (B, j, i)?? No.
    # Cost Matrix will be (B, N_start, N_end) where starts are 0..N-1, ends are 1..N?
    # Easier: (B, N+1, N+1).
    
    indices = torch.arange(N + 1, device=device)
    J = indices.view(1, N + 1, 1).expand(B, N + 1, N + 1)
    I = indices.view(1, 1, N + 1).expand(B, N + 1, N + 1)
    
    # Mask valid upper triangle: j < i
    mask_indices = J < I
    
    # Load Constraint
    # Load(j, i) = cum_load[i] - cum_load[j]
    loads = cum_load.unsqueeze(1) - cum_load.unsqueeze(2) # [B, N+1, 1] - [B, 1, N+1] -> [B, i, j] ?
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
    
    d0_pad = torch.cat([d_0_i, torch.zeros((B, 1), device=device)], dim=1) # (B, N+1)
    di0_pad = torch.cat([d_i_0, torch.zeros((B, 1), device=device)], dim=1) # (B, N+1)
    # Careful with indices.
    # d(0, T[j]): indices 0..N-1.
    # d(T[i-1], 0): indices 0..N-1.
    # i goes 1..N. i-1 goes 0..N-1.
    # j goes 0..N-1.
    
    # Matrix Broadcast
    # D_Start[j, i] depends only on j.
    D_Start = d0_pad.unsqueeze(2).expand(-1, -1, N+1) # (B, N+1, N+1) along j
    
    # D_End[j, i] depends only on i.
    # We need index i-1.
    # Shift d_i_0 right?
    # d_end_vec[i] = d_i_0[i-1].
    # d_end_vec[0] dummy.
    d_end_vec = torch.cat([torch.zeros((B,1), device=device), d_i_0], dim=1)
    D_End = d_end_vec.unsqueeze(1).expand(-1, N+1, -1) # (B, N+1, N+1) along i
    
    # Path Dist[j, i] = cum_dist[i] - cum_dist[j+1]
    # cum_dist indices: i maps to end, j+1 maps to start?
    # Verified in main func: path_dists = cum_dist_pad[i] - cum_dist_pad[j+1]
    # P[b, j, i] = cum_dist_pad[b, i] - cum_dist_pad[b, j+1]
    
    # We need cum_dist_pad shifted?
    # cum_dist_pad has N+1.
    # j+1 goes 1..N+1? if j=N, j+1=N+1 (out of bounds).
    # Pad extra?
    cd_pad = torch.cat([cum_dist_pad, torch.zeros((B, 1), device=device)], dim=1) # (B, N+2)
    
    # CD_I[b, j, i] = cum_dist_pad[b, i]
    # CD_J[b, j, i] = cd_pad[b, j+1]
    
    CD_I = cum_dist_pad.unsqueeze(1).expand(-1, N+1, -1)
    
    # For J+1, we shift/roll dim 1 of cum_dist_pad? Or just index?
    # Unfold?
    # Just construct vector: [cd[1], cd[2]... cd[N], 0].
    cd_j_vec = cd_pad[:, 1:N+2] # (B, N+1)
    CD_J = cd_j_vec.unsqueeze(2).expand(-1, -1, N+1)
    
    PathCosts = CD_I - CD_J
    
    Costs = D_Start + PathCosts + D_End
    Costs[~mask_cap] = float('inf')
    
    # Optimization Loop
    # V[k, i] = min cost to reach i using k vehicles.
    # Init
    V = torch.full((B, N+1), float('inf'), device=device)
    V[:, 0] = 0
    
    # Storage for reconstruction
    # P[k, i] = predecessor j
    P_k = torch.full((B, max_vehicles + 1, N + 1), -1, dtype=torch.long, device=device)
    
    best_cost = torch.full((B,), float('inf'), device=device)
    best_k_idx = torch.full((B,), -1, dtype=torch.long, device=device)
    
    # This might be memory hungry (B, N, N) for cost matrix.
    # With N=100, B=128 => 128*100*100 * 4 = 5MB. OK.
    
    for k in range(1, max_vehicles + 1):
        # V_new[i] = min_j (V_old[j] + Costs[j, i])
        # Broadcast V_old (B, j) -> (B, j, 1) -> (B, j, N+1)
        # Add to Costs (B, j, i)
        # Min over dim 1 (j)
        
        M = V.unsqueeze(2) + Costs
        min_vals, min_idxs = M.min(dim=1) # (B, i)
        
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
    routes_batch = []
    P_cpu = P_k.cpu().numpy()
    k_cpu = best_k.cpu().numpy()
    giant_tours_cpu = giant_tours.cpu().numpy()
    
    for b in range(B):
        k = k_cpu[b]
        if k == -1 or costs[b] == float('inf'):
            routes_batch.append([])
            continue
            
        curr = N
        route_nodes = []
        possible = True
        
        # Backtrack with known k
        current_k = k
        
        while curr > 0 and current_k > 0:
            prev = P_cpu[b, current_k, curr]
            if prev == -1:
                possible = False
                break
            
            nodes = giant_tours_cpu[b, prev:curr]
            route_nodes = [0] + list(nodes) + route_nodes
            
            curr = prev
            current_k -= 1
            
        if route_nodes and route_nodes[-1] != 0:
            route_nodes.append(0)
            
        routes_batch.append(route_nodes)
        
    return routes_batch, costs


# -----------------------------
# Vectorized Genetic Operators
# -----------------------------

def vectorized_ordered_crossover(parent1, parent2):
    """
    Vectorized Ordered Crossover (OX1) with shared cuts across batch.
    Args:
        parent1: (B, N)
        parent2: (B, N)
    Returns:
        offspring: (B, N)
    """
    B, N = parent1.size()
    device = parent1.device
    
    # 1. Generate shared cut points
    idx1, idx2 = torch.randint(0, N, (2,)).tolist()
    start = min(idx1, idx2)
    end = max(idx1, idx2)
    
    if start == end:
        end = min(start + 1, N)
        
    num_seg = end - start
    num_rem = N - num_seg
    
    # 2. Extract segment from Parent 1
    # segment: (B, num_seg)
    segment = parent1[:, start:end]
    
    # 3. Create Offspring container
    offspring = torch.zeros_like(parent1)
    # Copy segment
    offspring[:, start:end] = segment
    
    # 4. Fill remaining from Parent 2
    # We need to take elements from P2 that are NOT in segment, maintaining order
    # starting from 'end' index (wrapping around).
    
    # Roll P2 so it starts at 'end'
    # logical shift: indices [end, end+1, ..., N-1, 0, ..., end-1]
    roll_idx = torch.arange(N, device=device)
    roll_idx = (roll_idx + end) % N
    p2_rolled = parent2[:, roll_idx] # (B, N) sorted by fill order
    
    # Identification of valid elements
    # Since values are 1..N (or 0..N-1), we can use a mask.
    # Naive check: (B, N, 1) == (B, 1, num_seg) -> any(dim=2)
    # segment: (B, num_seg)
    # p2_rolled: (B, N)
    
    # Efficient exclusion check:
    # (B, N, 1) == (B, 1, num_seg) -> (B, N, num_seg) -> sum/any -> (B, N) mask
    # This uses B*N*num_seg memory. For N=100, B=128, this is ~1.2M elements (bool). Cheap.
    
    exists_in_seg = (p2_rolled.unsqueeze(2) == segment.unsqueeze(1)).any(dim=2) # (B, N)
    
    # We want elements where ~exists_in_seg
    # Since 'num_rem' is CONSTANT across batch, we can reshape.
    # valid_vals: (B, num_rem)
    # We select values where mask is False.
    # Since we know exactly num_rem values exist, we can use masking and reshape.
    valid_vals = p2_rolled[~exists_in_seg].view(B, num_rem)
    
    # 5. Place valid values into offspring
    # Determine indices to fill in offspring: [end...N-1, 0...start-1]
    fill_idx = torch.cat([
        torch.arange(end, N, device=device),
        torch.arange(0, start, device=device)
    ])
    
    offspring[:, fill_idx] = valid_vals
    
    return offspring


def calc_broken_pairs_distance(population):
    """
    Computes average Broken Pairs Distance for each individual in the population.
    Distance(A, B) = 1 - (|Edges(A) inter Edges(B)| / N)
    
    Args:
        population: (B, P, N) tensor of giant tours.
    Returns:
        diversity: (B, P) diversity score (higher is better/more distant).
    """
    B, P, N = population.size()
    device = population.device
    
    # 1. Construct Edge Hashes
    # Edges: (i, i+1) and (N-1, 0)
    # Hash: min(u,v)*N + max(u,v) (assuming max node index < N? No, nodes are 1..N usually, or indices 0..N-1?)
    # giant_tours usually contains indices.
    
    # Create cyclic view
    next_nodes = torch.roll(population, shifts=-1, dims=2)
    
    # Sort u,v to be direction agnostic
    u = torch.min(population, next_nodes)
    v = torch.max(population, next_nodes)
    
    # Hash (N_max is safely larger than N, e.g. N+1)
    # Be careful if nodes are indices (0..N-1) or IDs (1..N).
    # If 0..N-1, then max hash approx N^2.
    hashes = u * (N + 100) + v # (B, P, N)
    
    # 2. Compute Pairwise Distances
    # We want diversity[b, i] = mean_{j != i} (1 - intersection(i, j) / N)
    # Doing full PxP on GPU might be heavy if P is large (e.g. 100).
    # But for P=10-50, B=128, N=100:
    # (B, P, 1, N) == (B, 1, P, N) -> (B, P, P, N) comparison
    # Memory: 128 * 50 * 50 * 100 * 1 byte (bool) = 32 MB. Very Safe.
    
    # Expand for broadcast
    h_i = hashes.unsqueeze(2).unsqueeze(4) # (B, P, 1, N, 1)
    h_j = hashes.unsqueeze(1).unsqueeze(3) # (B, 1, P, 1, N)
    
    # Match matrix: matches[b, i, j, k, l]
    intersections = torch.zeros((B, P, P), device=device)
    for i in range(P):
        target = hashes[:, i:i+1, :]
        matches = (hashes.unsqueeze(3) == target.unsqueeze(2)) 
        num_shared = matches.any(dim=3).sum(dim=2) # (B, P)
        intersections[:, i, :] = num_shared
    
    # Distance = 1 - (intersection / N)
    dists = 1.0 - (intersections.float() / N)
    
    # Diversity of i = mean distance to others
    diversity = dists.sum(dim=2) / max(1, P - 1)
    return diversity


def vectorized_swap(tours, dist_matrix, max_iterations=200):
    """
    Vectorized Swap operator.
    """
    device = tours.device
    B, max_len = tours.shape
    
    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch: tours = tours.unsqueeze(0)
    
    # Handle dist matrix expansion
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    batch_indices = torch.arange(B, device=device).view(B, 1)

    for _ in range(max_iterations):
        # Sample indices for swap (i, j)
        # We sample random pairs instead of exhaustive search for speed
        idx = torch.randint(1, max_len - 1, (B, 2), device=device)
        i = torch.min(idx, dim=1)[0].view(B, 1)
        j = torch.max(idx, dim=1)[0].view(B, 1)
        
        # Ensure i != j and valid (not padding 0)
        # Check if nodes at i and j are not 0 (depot or padding)
        # Padding is usually 0 at end. Depot is 0 at start/end.
        # Indices 1..max_len-2 usually valid.
        
        node_i = torch.gather(tours, 1, i)
        node_j = torch.gather(tours, 1, j)
        
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any(): continue
        
        # Calculate Delta
        # i-1 -> i -> i+1
        # j-1 -> j -> j+1
        # Swap i and j.
        
        # Pre-swap neighbors
        prev_i_idx = i - 1
        next_i_idx = i + 1
        prev_j_idx = j - 1
        next_j_idx = j + 1
        
        # If adjacent (j = i+1), logic changes slightly
        # But for random sample, adjacent is just a special case computed correctly by cost?
        # A-B-C-D. Swap B,C -> A-C-B-D.
        # Edges removed: A-B, B-C, C-D. Added: A-C, C-B, B-D.
        
        # Special case adjacent:
        adjacent = (next_i_idx == j)
        
        # General case neighbors
        prev_i = torch.gather(tours, 1, prev_i_idx)
        next_i = torch.gather(tours, 1, next_i_idx)
        prev_j = torch.gather(tours, 1, prev_j_idx)
        next_j = torch.gather(tours, 1, next_j_idx)
        
        # Current edges cost
        d_curr  = dist_matrix[batch_indices, prev_i, node_i] + dist_matrix[batch_indices, node_i, next_i]
        d_curr += dist_matrix[batch_indices, prev_j, node_j] + dist_matrix[batch_indices, node_j, next_j]
        
        # New edges cost
        # Swap node_i and node_j locations
        # Edges: prev_i->node_j, node_j->next_i
        #        prev_j->node_i, node_i->next_j
        
        d_new  = dist_matrix[batch_indices, prev_i, node_j] + dist_matrix[batch_indices, node_j, next_i]
        d_new += dist_matrix[batch_indices, prev_j, node_i] + dist_matrix[batch_indices, node_i, next_j]
        
        # Correction for adjacent nodes: we double counted edge i-j (or B-C)
        # In current: B->C is in (node_i, next_i) AND (prev_j, node_j) if next_i==node_j
        # We subtracted B->C twice.
        # In new: C->B is in (node_j, next_i) AND (prev_j, node_i).
        # We added C->B twice.
        # So efficient delta:
        
        gain = d_curr - d_new
        
        # Apply improvement
        improved = (gain > 0.001) & mask
        
        if improved.any():
            # Apply swap
            # Update tours where improved
            # Need to reshape for scatter?
            
            # Create scatter indices
            idx_i = i[improved]
            idx_j = j[improved]
            
            # Values to swap from ORIGINAL tour
            val_i = node_i[improved]
            val_j = node_j[improved]
            
            # We want tours[b, idx_i] = val_j
            # tours[b, idx_j] = val_i
            # But improved is a mask over Batch.
            
            # Using scatter_. 
            # We need to construct src and index tensors for the whole batch or loop?
            # Mask indexing is easier.
            
            # tours[improved, i] = val_j -- This syntax tricky in torch for 2D.
            # tours[improved, i[improved]] works?
            # t = tours[improved] -> (K, max_len)
            # t.scatter_(1, idx_i, val_j)
            
            # Better:
            # We update batch-wise.
            batch_mask = improved.squeeze(1) # (B,)
            
            if batch_mask.any():
                sub_tours = tours[batch_mask]
                sub_i = i[batch_mask]
                sub_j = j[batch_mask]
                
                # Gather values again to be safe
                sub_val_i = torch.gather(sub_tours, 1, sub_i)
                sub_val_j = torch.gather(sub_tours, 1, sub_j)
                
                # Scatter swap
                sub_tours.scatter_(1, sub_i, sub_val_j)
                sub_tours.scatter_(1, sub_j, sub_val_i)
                
                tours[batch_mask] = sub_tours

    return tours


def vectorized_relocate(tours, dist_matrix, max_iterations=200):
    """
    Vectorized Relocate operator. Moves a node to a new position.
    """
    device = tours.device
    B, max_len = tours.shape
    
    batch_indices = torch.arange(B, device=device).view(B, 1)
    
    # Expand dist matrix if needed
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # Sample source idx (to move) and dest idx (insertion point)
        # We want to move node at i to insert after j
        idx = torch.randint(1, max_len - 1, (B, 2), device=device)
        i = idx[:, 0:1]
        j = idx[:, 1:2]
        
        # Valid checks: node_i != 0, i != j, i != j+1 (move to same spot)
        node_i = torch.gather(tours, 1, i)
        
        # Mask valid i
        mask = (node_i != 0) & (i != j) & (i != j + 1)
        if not mask.any(): continue
        
        # Calculate Cost Change
        # Remove i: prev_i -> i -> next_i.  => prev_i -> next_i
        # Insert after j: j -> next_j. => j -> i -> next_j
        
        prev_i_idx = i - 1
        next_i_idx = i + 1
        
        next_j_idx = j + 1 # Insert after j
        # But if we remove i, indices shift?
        # Relocate logic on array implies shift.
        # Vectorized shift is expensive (requires creating new tensor).
        # Swap-based chains? Relocate is hard to vectorize fully in-place.
        
        # Alternative: We only do it if valid.
        
        # Let's assess cost first.
        prev_i = torch.gather(tours, 1, prev_i_idx)
        next_i = torch.gather(tours, 1, next_i_idx)
        
        node_j = torch.gather(tours, 1, j)
        next_j = torch.gather(tours, 1, next_j_idx) # Could be 0 (end of route), which is valid
        
        # Removal cost change
        d_remove = - dist_matrix[batch_indices, prev_i, node_i] \
                   - dist_matrix[batch_indices, node_i, next_i] \
                   + dist_matrix[batch_indices, prev_i, next_i]
                   
        # Insertion cost change
        d_insert = - dist_matrix[batch_indices, node_j, next_j] \
                   + dist_matrix[batch_indices, node_j, node_i] \
                   + dist_matrix[batch_indices, node_i, next_j]
                   
        gain = -(d_remove + d_insert) # Gain = -Delta
        
        improved = (gain > 0.001) & mask
        
        if improved.any():
            # Apply relocate by rewriting the row
            # Expensive part: we need to slice and concat for each row?
            # Or mask based generic reconstruction.
            # Only do it for improved batch items.
            
            batch_mask = improved.squeeze(1)
            b_indices = torch.nonzero(batch_mask).squeeze(1)
            
            for b_idx in b_indices:
                tour = tours[b_idx]
                val_i = i[b_idx].item()
                val_j = j[b_idx].item()
                
                # Manual shift for specific row (CPU or GPU op)
                # GPU slice concat
                # Remove i
                node_val = tour[val_i]
                # Concat [0..i) and (i+1..end)
                rem_tour = torch.cat([tour[:val_i], tour[val_i+1:], torch.tensor([0], device=device)])
                
                # Insert after j
                # If j > i, index j has shifted down by 1?
                # Logic: j is index in ORIGINAL tour.
                # If j < i: index j is same. Insert at j+1.
                # If j > i: index j becomes j-1. Insert at j.
                
                eff_j = val_j
                if val_j > val_i:
                    eff_j -= 1
                    
                # Insert at eff_j + 1
                final_tour = torch.cat([rem_tour[:eff_j+1], node_val.view(1), rem_tour[eff_j+1:-1]])
                # Ensure length matches (we appended 0 earlier, removed one)
                
                tours[b_idx] = final_tour

    return tours


def vectorized_two_opt_star(tours, dist_matrix, max_iterations=200):
    """
    Vectorized 2-opt* operator. (Tail Swap).
    Exchanges tails of two different routes.
    """
    device = tours.device
    B, max_len = tours.shape
    
    batch_indices = torch.arange(B, device=device).view(B, 1)
    
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)
        
    for _ in range(max_iterations):
        # Sample i, j
        idx = torch.randint(1, max_len - 1, (B, 2), device=device)
        i = idx[:, 0:1]
        j = idx[:, 1:2]
        
        node_i = torch.gather(tours, 1, i)
        node_j = torch.gather(tours, 1, j)
        
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any(): continue
        
        # Check routes: are they separated by a 0?
        # We need "tail" locations.
        is_zero = (tours == 0)
        seq = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
        
        mask_i = seq > i 
        possible_i = is_zero & mask_i
        end_i = torch.argmax(possible_i.float(), dim=1).view(B, 1) 
        
        mask_j = seq > j
        possible_j = is_zero & mask_j
        end_j = torch.argmax(possible_j.float(), dim=1).view(B, 1)
        
        valid_search = (end_i > i) & (end_j > j)
        mask = mask & valid_search
        
        inter_route = (end_i != end_j)
        mask = mask & inter_route
        if not mask.any(): continue
        
        # Calculate Delta
        # New R1: ... u -> v_next ... 0
        # New R2: ... v -> u_next ... 0
        # Edges broken: (u, u_next), (v, v_next)
        # Edges added: (u, v_next), (v, u_next)
        
        u = node_i
        v = node_j
        
        next_i_idx = i + 1
        next_j_idx = j + 1
        
        u_next = torch.gather(tours, 1, next_i_idx)
        v_next = torch.gather(tours, 1, next_j_idx)
        
        d_curr  = dist_matrix[batch_indices, u, u_next] + dist_matrix[batch_indices, v, v_next]
        d_new   = dist_matrix[batch_indices, u, v_next] + dist_matrix[batch_indices, v, u_next]
        
        gain = d_curr - d_new
        improved = (gain > 0.001) & mask
        
        if improved.any():
            batch_mask = improved.squeeze(1)
            b_indices = torch.nonzero(batch_mask).squeeze(1)
            
            for b_idx in b_indices:
                tour = tours[b_idx]
                idx_i = i[b_idx].item()
                idx_j = j[b_idx].item()
                e_i = end_i[b_idx].item()
                e_j = end_j[b_idx].item()
                
                # Segments
                # Route 1: [start_1 ... i] + [j+1 ... e_j] + 0
                # Route 2: [start_2 ... j] + [i+1 ... e_i] + 0
                
                if e_i < idx_j:
                    p1 = tour[:idx_i+1]
                    p2 = tour[idx_j+1 : e_j]
                    p3 = tour[e_i : idx_j+1]
                    p4 = tour[idx_i+1 : e_i]
                    p5 = tour[e_j:]
                    new_tour = torch.cat([p1, p2, p3, p4, p5])
                    
                elif e_j < idx_i:
                    p1 = tour[:idx_j+1]
                    p2 = tour[idx_i+1 : e_i]
                    p3 = tour[e_j : idx_i+1]
                    p4 = tour[idx_j+1 : e_j]
                    p5 = tour[e_i:]
                    new_tour = torch.cat([p1, p2, p3, p4, p5])
                    
                else: 
                     continue
                
                tours[b_idx] = new_tour
                
    return tours


def vectorized_swap_star(tours, dist_matrix, max_iterations=100):
    """
    Vectorized Swap* operator.
    Exchanges u and v between different routes, re-inserting them at best positions.
    """
    device = tours.device
    B, max_len = tours.shape
    
    batch_indices = torch.arange(B, device=device).view(B, 1)
    seq = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
    
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)
        
    for _ in range(max_iterations):
        # 1. Sample u (i) and v (j)
        idx = torch.randint(1, max_len - 1, (B, 2), device=device)
        i = idx[:, 0:1]
        j = idx[:, 1:2]
        
        node_i = torch.gather(tours, 1, i)
        node_j = torch.gather(tours, 1, j)
        
        mask = (node_i != 0) & (node_j != 0) & (i != j)
        if not mask.any(): continue
        
        # 2. Identify Routes
        is_zero = (tours == 0)
        
        mask_after_i = seq > i
        poss_end_i = is_zero & mask_after_i
        end_i = torch.argmax(poss_end_i.float(), dim=1).view(B, 1)
        
        mask_before_i = seq < i
        valid_start_i = is_zero & mask_before_i
        start_i = torch.max(torch.where(valid_start_i, seq, -1), dim=1)[0].view(B, 1) 
        
        mask_after_j = seq > j
        poss_end_j = is_zero & mask_after_j
        end_j = torch.argmax(poss_end_j.float(), dim=1).view(B, 1)
        
        mask_before_j = seq < j
        valid_start_j = is_zero & mask_before_j
        start_j = torch.max(torch.where(valid_start_j, seq, -1), dim=1)[0].view(B, 1)
        
        valid_bounds = (end_i > i) & (start_i < i) & (end_j > j) & (start_j < j) & (start_i >= 0) & (start_j >= 0)
        mask = mask & valid_bounds
        
        inter_route = (start_i != start_j) 
        mask = mask & inter_route
        if not mask.any(): continue
        
        # 3. Compute Removal Gains
        prev_i_node = torch.gather(tours, 1, i - 1)
        next_i_node = torch.gather(tours, 1, i + 1)
        rem_gain_i = dist_matrix[batch_indices, prev_i_node, node_i] + \
                     dist_matrix[batch_indices, node_i, next_i_node] - \
                     dist_matrix[batch_indices, prev_i_node, next_i_node]
                     
        prev_j_node = torch.gather(tours, 1, j - 1)
        next_j_node = torch.gather(tours, 1, j + 1)
        rem_gain_j = dist_matrix[batch_indices, prev_j_node, node_j] + \
                     dist_matrix[batch_indices, node_j, next_j_node] - \
                     dist_matrix[batch_indices, prev_j_node, next_j_node]
                     
        # 4. Find Best Insertion
        next_nodes = torch.roll(tours, shifts=-1, dims=1)
        u_exp = node_i.expand(-1, max_len)
        v_exp = node_j.expand(-1, max_len)
        
        batch_rows = batch_indices.expand(-1, max_len)
        
        d_k_u = dist_matrix[batch_rows, tours, u_exp]
        d_u_next = dist_matrix[batch_rows, u_exp, next_nodes]
        d_k_next = dist_matrix[batch_rows, tours, next_nodes]
        
        insert_cost_u = d_k_u + d_u_next - d_k_next
        
        mask_J = (seq >= start_j) & (seq < end_j) & (seq != j) & (seq != j-1)
        insert_cost_u_masked = torch.where(mask_J, insert_cost_u, torch.tensor(float('inf'), device=device))
        best_ins_u_val, best_ins_u_idx = torch.min(insert_cost_u_masked, dim=1)
        best_ins_u_val, best_ins_u_idx = best_ins_u_val.view(B, 1), best_ins_u_idx.view(B, 1)
        
        d_k_v = dist_matrix[batch_rows, tours, v_exp]
        d_v_next = dist_matrix[batch_rows, v_exp, next_nodes]
        insert_cost_v = d_k_v + d_v_next - d_k_next
        
        mask_I = (seq >= start_i) & (seq < end_i) & (seq != i) & (seq != i-1)
        insert_cost_v_masked = torch.where(mask_I, insert_cost_v, torch.tensor(float('inf'), device=device))
        best_ins_v_val, best_ins_v_idx = torch.min(insert_cost_v_masked, dim=1)
        best_ins_v_val, best_ins_v_idx = best_ins_v_val.view(B, 1), best_ins_v_idx.view(B, 1)
        
        total_gain = rem_gain_i + rem_gain_j - best_ins_u_val - best_ins_v_val
        improved = (total_gain > 0.001) & mask
        
        if improved.any():
            # Apply moves
            batch_mask = improved.squeeze(1)
            b_indices = torch.nonzero(batch_mask).squeeze(1)
            
            for b_idx in b_indices:
                tour = tours[b_idx]
                
                pos_i, pos_j = i[b_idx].item(), j[b_idx].item()
                ins_u, ins_v = best_ins_u_idx[b_idx].item(), best_ins_v_idx[b_idx].item()
                val_u, val_v = node_i[b_idx].item(), node_j[b_idx].item()
                
                t_list = tour.tolist()
                
                tgt_u, tgt_v = ins_u + 1, ins_v + 1
                
                first_rem, second_rem = min(pos_i, pos_j), max(pos_i, pos_j)
                
                if first_rem < tgt_u: tgt_u -= 1
                if second_rem < tgt_u: tgt_u -= 1
                
                if first_rem < tgt_v: tgt_v -= 1
                if second_rem < tgt_v: tgt_v -= 1
                
                new_t = [x for k, x in enumerate(t_list) if k != pos_i and k != pos_j]
                
                if tgt_u > tgt_v:
                    new_t.insert(tgt_u, val_u)
                    new_t.insert(tgt_v, val_v)
                else:
                    new_t.insert(tgt_v, val_v)
                    new_t.insert(tgt_u, val_u)
                
                tours[b_idx] = torch.tensor(new_t, device=device)

    return tours






class VectorizedPopulation:
    def __init__(self, size, device, alpha_diversity=0.5):
        self.max_size = size
        self.device = device
        self.alpha_diversity = alpha_diversity
        self.population = None # (B, P, N)
        self.costs = None # (B, P)
        self.biased_fitness = None # (B, P)
        self.diversity_scores = None # (B, P)
        
    def initialize(self, initial_pop, initial_costs):
        """
        Args:
            initial_pop: (B, N) or (B, P0, N)
            initial_costs: (B,) or (B, P0)
        """
        if initial_pop.dim() == 2:
            initial_pop = initial_pop.unsqueeze(1) # (B, 1, N)
        
        # Consistent costs shape (B, P0)
        if initial_costs.dim() == 1:
            initial_costs = initial_costs.unsqueeze(1) # (B, 1)
            
        self.population = initial_pop
        self.costs = initial_costs
        self.compute_biased_fitness()
        
    def add_individuals(self, candidates, costs):
        """
        Merge new individuals and select survivors.
        Args:
            candidates: (B, C, N)
            costs: (B, C)
        """
        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(1)
            costs = costs.unsqueeze(1)
            
        # 1. Concatenate
        combined_pop = torch.cat([self.population, candidates], dim=1) # (B, P+C, N)
        combined_costs = torch.cat([self.costs, costs], dim=1) # (B, P+C)
        
        # 2. Update state temporarily
        self.population = combined_pop
        self.costs = combined_costs
        
        # 3. Compute Fitness & Survivor Selection
        self.compute_biased_fitness()
        
        # Select best P based on biased fitness
        # We want smallest biased_fitness (rank sum)
        
        if self.population.size(1) > self.max_size:
            # Sort by fitness (ascending, smaller is better)
            _, indices = torch.sort(self.biased_fitness, dim=1)
            survivors = indices[:, :self.max_size] # (B, max_size)
            
            # Gather survivors
            B, _, N = self.population.size()
            
            # Gather population
            surv_expanded = survivors.unsqueeze(2).expand(-1, -1, N)
            self.population = torch.gather(self.population, 1, surv_expanded)
            
            # Gather costs
            self.costs = torch.gather(self.costs, 1, survivors)
            
            # Gather fitness
            self.biased_fitness = torch.gather(self.biased_fitness, 1, survivors)
            
            # Gather diversity (optional)
            if self.diversity_scores is not None:
                self.diversity_scores = torch.gather(self.diversity_scores, 1, survivors)

    def compute_biased_fitness(self):
        """
        Compute Biased Fitness = Rank(Cost) + alpha * Rank(Diversity)
        Lower is better for both ranks (0 is best).
        """
        B, P, N = self.population.size()
        
        # 1. Rank by Cost (Ascending: lower cost is better, Rank 0)
        cost_indices = torch.argsort(self.costs, dim=1)
        cost_ranks = torch.argsort(cost_indices, dim=1).float() 
        
        # 2. Diversity
        # Calculate diversity (higher is better)
        self.diversity_scores = calc_broken_pairs_distance(self.population)
        
        # Rank by Diversity (Descending: higher diversity is better, Rank 0)
        div_indices = torch.argsort(self.diversity_scores, dim=1, descending=True)
        div_ranks = torch.argsort(div_indices, dim=1).float()
        
        # 3. Combine
        # Fitness = CostRank + alpha * DiversityRank
        self.biased_fitness = cost_ranks + self.alpha_diversity * div_ranks

    def get_parents(self, n_offspring=1):
        """
        Binary tournament selection.
        Returns: parents1 (B, n_offspring, N), parents2 (B, n_offspring, N)
        """
        B, P, N = self.population.size()
        
        def tournament():
            idx_a = torch.randint(0, P, (B, n_offspring), device=self.device)
            idx_b = torch.randint(0, P, (B, n_offspring), device=self.device)
            fit_a = torch.gather(self.biased_fitness, 1, idx_a)
            fit_b = torch.gather(self.biased_fitness, 1, idx_b)
            return torch.where(fit_a < fit_b, idx_a, idx_b)
            
        parent1_idx = tournament()
        parent2_idx = tournament()
        
        p1 = torch.gather(self.population, 1, parent1_idx.unsqueeze(2).expand(-1,-1,N))
        p2 = torch.gather(self.population, 1, parent2_idx.unsqueeze(2).expand(-1,-1,N))
        
        return p1, p2

class VectorizedHGS:
    def __init__(self, dist_matrix, demands, vehicle_capacity, time_limit=1.0, device='cuda'):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.time_limit = time_limit
        self.device = device
        
    def solve(self, initial_solutions, n_generations=50, population_size=10, elite_size=5, time_limit=None, max_vehicles=0):
        """
        Run HGS starting from initial solutions (Expert Imitation Mode).
        """
        B, N = initial_solutions.size()
        start_time = time.time()
        
        # Initial Evaluation
        _, costs = vectorized_linear_split(
            initial_solutions, self.dist_matrix, self.demands, self.vehicle_capacity, max_vehicles=max_vehicles
        )
        
        pop = VectorizedPopulation(population_size, self.device)
        pop.initialize(initial_solutions, costs)
        
        no_improv = 0
        best_cost_tracker = pop.costs.min().item()
        
        for gen in range(n_generations):
            if time_limit is not None and (time.time() - start_time > time_limit):
                break
                
            # Selection
            p1, p2 = pop.get_parents(n_offspring=1) 
            p1 = p1.squeeze(1)
            p2 = p2.squeeze(1)
            
            # Crossover
            offspring_giant = vectorized_ordered_crossover(p1, p2)
            
            # Evaluation
            routes_list, split_costs = vectorized_linear_split(
                offspring_giant, self.dist_matrix, self.demands, self.vehicle_capacity, max_vehicles=max_vehicles
            )
            
            # Education (Local Search)
            max_l = max(len(r) for r in routes_list)
            offspring_routes = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
            for b in range(B):
                r = routes_list[b]
                offspring_routes[b, :len(r)] = torch.tensor(r, device=self.device)
            
            if max_l > 2:
                improved_routes = local_search_2opt_vectorized(offspring_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_swap(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_relocate(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_two_opt_star(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_swap_star(improved_routes, self.dist_matrix, max_iterations=50)
                
                # Recalculate cost
                from_n = improved_routes[:, :-1]
                to_n = improved_routes[:, 1:]
                
                if self.dist_matrix.dim() == 3 and self.dist_matrix.size(0) == B:
                    batch_ids = torch.arange(B, device=self.device).view(B, 1)
                    dists = self.dist_matrix[batch_ids, from_n, to_n]
                else:
                    expanded_dm = self.dist_matrix
                    if expanded_dm.dim() == 2:
                        expanded_dm = expanded_dm.unsqueeze(0)
                    if expanded_dm.size(0) == 1 and B > 1:
                         expanded_dm = expanded_dm.expand(B, -1, -1)
                    batch_ids = torch.arange(B, device=self.device).view(B, 1)
                    dists = expanded_dm[batch_ids, from_n, to_n]
                    
                improved_costs = dists.sum(dim=1)
            else:
                improved_routes = offspring_routes
                improved_costs = split_costs
            
            # Reconstruct Giant Tour
            giant_candidates = torch.zeros((B, N), dtype=torch.long, device=self.device)
            improved_routes_cpu = improved_routes.cpu().numpy()
            
            for b in range(B):
                r_full = improved_routes_cpu[b]
                g_tour = r_full[r_full != 0]
                if len(g_tour) == N:
                    giant_candidates[b, :] = torch.tensor(g_tour, device=self.device)
                else:
                    giant_candidates[b, :] = offspring_giant[b, :]

            # Survivor Selection
            pop.add_individuals(giant_candidates, improved_costs)
            
            # Stagnation Check
            current_best = pop.costs.min().item()
            if current_best < best_cost_tracker - 1e-4:
                best_cost_tracker = current_best
                no_improv = 0
            else:
                no_improv += 1
                
            if no_improv > 50:
                 k = elite_size
                 # Keep Elite
                 sorted_idx = torch.argsort(pop.biased_fitness, dim=1)[:, :k]
                 elite_pop = torch.gather(pop.population, 1, sorted_idx.unsqueeze(2).expand(-1, -1, N))
                 elite_cost = torch.gather(pop.costs, 1, sorted_idx)
                 
                 # New candidates (Random Permutations)
                 n_rest = pop.max_size - k
                 new_pop = torch.zeros((B, n_rest, N), dtype=torch.long, device=self.device)
                 for b in range(B):
                     for i in range(n_rest):
                         new_pop[b, i] = torch.randperm(N, device=self.device) + 1 
                         
                 # Eval new pop
                 # Flatten (B, n_rest, N) -> (B*n_rest, N)
                 b_sz, n_rst, n_nds = new_pop.shape
                 flat_pop = new_pop.view(b_sz * n_rst, n_nds)
                 
                 flat_dist = self.dist_matrix.repeat_interleave(n_rst, dim=0)
                 flat_demands = self.demands.repeat_interleave(n_rst, dim=0)
                 flat_cap = self.vehicle_capacity
                 if isinstance(flat_cap, torch.Tensor) and flat_cap.dim() > 0:
                     flat_cap = flat_cap.repeat_interleave(n_rst, dim=0)
                     
                 _, new_costs_flat = vectorized_linear_split(
                     flat_pop, flat_dist, flat_demands, flat_cap, max_vehicles=max_vehicles
                 )
                 new_costs = new_costs_flat.view(b_sz, n_rst)
                 
                 # Update pop directly
                 pop.population = torch.cat([elite_pop, new_pop], dim=1)
                 pop.costs = torch.cat([elite_cost, new_costs], dim=1)
                 pop.compute_biased_fitness()
                 no_improv = 0
            
        best_cost, best_idx = torch.min(pop.costs, dim=1)
        best_giant = pop.population[torch.arange(B), best_idx] 
        best_routes, _ = vectorized_linear_split(
            best_giant, self.dist_matrix, self.demands, self.vehicle_capacity, max_vehicles=max_vehicles
        )
        return best_routes, best_cost
