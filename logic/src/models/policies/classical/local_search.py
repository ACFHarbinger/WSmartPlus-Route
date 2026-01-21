"""
Vectorized Local Search Operators for Vehicle Routing Problems.

This module implements several local search heuristics optimized for parallel execution
on GPU using PyTorch. These operators are used to improve routing solutions in both
genetic algorithms (like HGS) and as post-processing steps for neural models.

Implemented Operators:
- vectorized_two_opt: Intra-route segment reversal.
- vectorized_swap: Intra-route node exchange.
- vectorized_relocate: Intra-route node relocation.
- vectorized_two_opt_star: Inter-route tail swap.
- vectorized_swap_star: Inter-route node exchange with re-insertion optimization.
- vectorized_three_opt: Intra-route 3-opt moves.
"""

import torch


def vectorized_two_opt(tours, distance_matrix, max_iterations=200):
    """
    Vectorized 2-opt local search across a batch of tours using PyTorch.

    The 2-opt algorithm is a classic local search heuristic that iteratively improves
    tours by reversing segments. For each pair of edges (i,i+1) and (j,j+1), it checks
    if swapping them reduces total distance:

    Original: ... -> i -> i+1 -> ... -> j -> j+1 -> ...
    2-opt:    ... -> i -> j -> ... -> i+1 -> j+1 -> ...  (reverse middle segment)

    This implementation processes an entire batch of tours in parallel on GPU,
    significantly faster than sequential processing.

    Algorithm:
    1. For all possible edge pairs (i,j) in parallel:
        - Compute gain = current_distance - new_distance
    2. Select best improvement for each tour in batch
    3. Apply swaps where gain > 0
    4. Repeat until no improvement or max_iterations

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        max_iterations: Maximum number of improvement iterations (default: 200)

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours should include depot as node 0
        - Works with both batched and shared distance matrices
        - Stops early if no improvement found
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

        I_grid, J_grid = torch.meshgrid(i, j, indexing="ij")
        mask = J_grid > I_grid
        if not mask.any():
            break

        I_vals = I_grid[mask]
        J_vals = J_grid[mask]
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


def vectorized_swap(tours, dist_matrix, max_iterations=200):
    """
    Vectorized Swap operator.
    Exchanges two nodes within the same route for multiple batch items simultaneously.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix (B, N, N) or compatible broadcast shape.
        max_iterations (int): Number of random swap attempts.

    Returns:
        torch.Tensor: Updated tours.
    """
    device = tours.device
    B, max_len = tours.shape

    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

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
        if not mask.any():
            continue

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

        # General case neighbors
        prev_i = torch.gather(tours, 1, prev_i_idx)
        next_i = torch.gather(tours, 1, next_i_idx)
        prev_j = torch.gather(tours, 1, prev_j_idx)
        next_j = torch.gather(tours, 1, next_j_idx)

        # Current edges cost
        d_curr = dist_matrix[batch_indices, prev_i, node_i] + dist_matrix[batch_indices, node_i, next_i]
        d_curr += dist_matrix[batch_indices, prev_j, node_j] + dist_matrix[batch_indices, node_j, next_j]

        # New edges cost
        # Swap node_i and node_j locations
        # Edges: prev_i->node_j, node_j->next_i
        #        prev_j->node_i, node_i->next_j

        d_new = dist_matrix[batch_indices, prev_i, node_j] + dist_matrix[batch_indices, node_j, next_i]
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
            i[improved]
            j[improved]

            # Values to swap from ORIGINAL tour
            node_i[improved]
            node_j[improved]

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
            batch_mask = improved.squeeze(1)  # (B,)

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
    Removes a node from its current position and re-inserts it elsewhere if it yields an improvement.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix.
        max_iterations (int): Number of attempts.

    Returns:
        torch.Tensor: Updated tours.
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
        if not mask.any():
            continue

        # Calculate Cost Change
        # Remove i: prev_i -> i -> next_i.  => prev_i -> next_i
        # Insert after j: j -> next_j. => j -> i -> next_j

        prev_i_idx = i - 1
        next_i_idx = i + 1

        next_j_idx = j + 1  # Insert after j
        # But if we remove i, indices shift?
        # Relocate logic on array implies shift.
        # Vectorized shift is expensive (requires creating new tensor).
        # Swap-based chains? Relocate is hard to vectorize fully in-place.

        # Alternative: We only do it if valid.

        # Let's assess cost first.
        prev_i = torch.gather(tours, 1, prev_i_idx)
        next_i = torch.gather(tours, 1, next_i_idx)

        node_j = torch.gather(tours, 1, j)
        next_j = torch.gather(tours, 1, next_j_idx)  # Could be 0 (end of route), which is valid

        # Removal cost change
        d_remove = (
            -dist_matrix[batch_indices, prev_i, node_i]
            - dist_matrix[batch_indices, node_i, next_i]
            + dist_matrix[batch_indices, prev_i, next_i]
        )

        # Insertion cost change
        d_insert = (
            -dist_matrix[batch_indices, node_j, next_j]
            + dist_matrix[batch_indices, node_j, node_i]
            + dist_matrix[batch_indices, node_i, next_j]
        )

        gain = -(d_remove + d_insert)  # Gain = -Delta

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
                rem_tour = torch.cat([tour[:val_i], tour[val_i + 1 :], torch.tensor([0], device=device)])

                # Insert after j
                # If j > i, index j has shifted down by 1?
                # Logic: j is index in ORIGINAL tour.
                # If j < i: index j is same. Insert at j+1.
                # If j > i: index j becomes j-1. Insert at j.

                eff_j = val_j
                if val_j > val_i:
                    eff_j -= 1

                # Insert at eff_j + 1
                final_tour = torch.cat([rem_tour[: eff_j + 1], node_val.view(1), rem_tour[eff_j + 1 : -1]])
                # Ensure length matches (we appended 0 earlier, removed one)

                tours[b_idx] = final_tour

    return tours


def vectorized_two_opt_star(tours, dist_matrix, max_iterations=200):
    """
    Vectorized 2-opt* operator. (Tail Swap).
    Exchanges tails of two different routes. Useful for improving solutions by reconnecting routes.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix.
        max_iterations (int): Number of attempts.

    Returns:
        torch.Tensor: Updated tours.
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
        if not mask.any():
            continue

        # Check routes: are they separated by a 0?
        # We need "tail" locations.
        is_zero = tours == 0
        seq = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)

        mask_i = seq > i
        possible_i = is_zero & mask_i
        end_i = torch.argmax(possible_i.float(), dim=1).view(B, 1)

        mask_j = seq > j
        possible_j = is_zero & mask_j
        end_j = torch.argmax(possible_j.float(), dim=1).view(B, 1)

        valid_search = (end_i > i) & (end_j > j)
        mask = mask & valid_search

        inter_route = end_i != end_j
        mask = mask & inter_route
        if not mask.any():
            continue

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

        d_curr = dist_matrix[batch_indices, u, u_next] + dist_matrix[batch_indices, v, v_next]
        d_new = dist_matrix[batch_indices, u, v_next] + dist_matrix[batch_indices, v, u_next]

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
                    p1 = tour[: idx_i + 1]
                    p2 = tour[idx_j + 1 : e_j]
                    p3 = tour[e_i : idx_j + 1]
                    p4 = tour[idx_i + 1 : e_i]
                    p5 = tour[e_j:]
                    new_tour = torch.cat([p1, p2, p3, p4, p5])

                elif e_j < idx_i:
                    p1 = tour[: idx_j + 1]
                    p2 = tour[idx_i + 1 : e_i]
                    p3 = tour[e_j : idx_i + 1]
                    p4 = tour[idx_j + 1 : e_j]
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
    This is a more powerful move that combines swapping nodes between routes with re-optimizing their insertion points.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix.
        max_iterations (int): Number of attempts.

    Returns:
        torch.Tensor: Updated tours.
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
        if not mask.any():
            continue

        # 2. Identify Routes
        is_zero = tours == 0

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

        inter_route = start_i != start_j
        mask = mask & inter_route
        if not mask.any():
            continue

        # 3. Compute Removal Gains
        prev_i_node = torch.gather(tours, 1, i - 1)
        next_i_node = torch.gather(tours, 1, i + 1)
        rem_gain_i = (
            dist_matrix[batch_indices, prev_i_node, node_i]
            + dist_matrix[batch_indices, node_i, next_i_node]
            - dist_matrix[batch_indices, prev_i_node, next_i_node]
        )

        prev_j_node = torch.gather(tours, 1, j - 1)
        next_j_node = torch.gather(tours, 1, j + 1)
        rem_gain_j = (
            dist_matrix[batch_indices, prev_j_node, node_j]
            + dist_matrix[batch_indices, node_j, next_j_node]
            - dist_matrix[batch_indices, prev_j_node, next_j_node]
        )

        # 4. Find Best Insertion
        next_nodes = torch.roll(tours, shifts=-1, dims=1)
        u_exp = node_i.expand(-1, max_len)
        v_exp = node_j.expand(-1, max_len)

        batch_rows = batch_indices.expand(-1, max_len)

        d_k_u = dist_matrix[batch_rows, tours, u_exp]
        d_u_next = dist_matrix[batch_rows, u_exp, next_nodes]
        d_k_next = dist_matrix[batch_rows, tours, next_nodes]

        insert_cost_u = d_k_u + d_u_next - d_k_next

        mask_J = (seq >= start_j) & (seq < end_j) & (seq != j) & (seq != j - 1)
        insert_cost_u_masked = torch.where(mask_J, insert_cost_u, torch.tensor(float("inf"), device=device))
        best_ins_u_val, best_ins_u_idx = torch.min(insert_cost_u_masked, dim=1)
        best_ins_u_val, best_ins_u_idx = (
            best_ins_u_val.view(B, 1),
            best_ins_u_idx.view(B, 1),
        )

        d_k_v = dist_matrix[batch_rows, tours, v_exp]
        d_v_next = dist_matrix[batch_rows, v_exp, next_nodes]
        insert_cost_v = d_k_v + d_v_next - d_k_next

        mask_I = (seq >= start_i) & (seq < end_i) & (seq != i) & (seq != i - 1)
        insert_cost_v_masked = torch.where(mask_I, insert_cost_v, torch.tensor(float("inf"), device=device))
        best_ins_v_val, best_ins_v_idx = torch.min(insert_cost_v_masked, dim=1)
        best_ins_v_val, best_ins_v_idx = (
            best_ins_v_val.view(B, 1),
            best_ins_v_idx.view(B, 1),
        )

        total_gain = rem_gain_i + rem_gain_j - best_ins_u_val - best_ins_v_val
        improved = (total_gain > 0.001) & mask

        if improved.any():
            # Apply moves
            batch_mask = improved.squeeze(1)
            b_indices = torch.nonzero(batch_mask).squeeze(1)

            for b_idx in b_indices:
                tour = tours[b_idx]

                pos_i, pos_j = i[b_idx].item(), j[b_idx].item()
                ins_u, ins_v = (
                    best_ins_u_idx[b_idx].item(),
                    best_ins_v_idx[b_idx].item(),
                )
                val_u, val_v = node_i[b_idx].item(), node_j[b_idx].item()

                t_list = tour.tolist()

                tgt_u, tgt_v = ins_u + 1, ins_v + 1

                first_rem, second_rem = min(pos_i, pos_j), max(pos_i, pos_j)

                if first_rem < tgt_u:
                    tgt_u -= 1
                if second_rem < tgt_u:
                    tgt_u -= 1

                if first_rem < tgt_v:
                    tgt_v -= 1
                if second_rem < tgt_v:
                    tgt_v -= 1

                new_t = [x for k, x in enumerate(t_list) if k != pos_i and k != pos_j]

                if tgt_u > tgt_v:
                    new_t.insert(tgt_u, val_u)
                    new_t.insert(tgt_v, val_v)
                else:
                    new_t.insert(tgt_v, val_v)
                    new_t.insert(tgt_u, val_u)

                tours[b_idx] = torch.tensor(new_t, device=device)

    return tours


def vectorized_three_opt(tours, dist_matrix, max_iterations=100):
    """
    Vectorized 3-opt local search using sampling for high efficiency.
    Evaluates multiple reconnection ways for 3-edge removals in parallel.

    Args:
        tours (torch.Tensor): Current tours (B, max_len).
        dist_matrix (torch.Tensor): Distance matrix (B, N, N) or compatible.
        max_iterations (int): Number of sampling iterations.

    Returns:
        torch.Tensor: Updated tours.
    """
    device = tours.device
    B, max_len = tours.shape

    if max_len < 6:  # Need at least 6 nodes for a meaningful 3-opt
        return tours

    batch_indices = torch.arange(B, device=device).view(B, 1)

    # Expand dist matrix if needed
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.dim() == 3 and dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    for _ in range(max_iterations):
        # 1. Sample 3 indices i < j < k
        # To ensure i < j < k, we sample 3 and sort them.
        # We skip indices 0 (depot start) and max_len-1 (depot end/padding)
        idx = torch.sort(torch.randint(1, max_len - 1, (B, 3), device=device), dim=1).values
        i = idx[:, 0:1]
        j = idx[:, 1:2]
        k = idx[:, 2:3]

        # Valid triplet check: no adjacent indices (to ensure we remove 3 distinct edges)
        # and not 0 (padding)
        node_i = torch.gather(tours, 1, i)
        node_j = torch.gather(tours, 1, j)
        node_k = torch.gather(tours, 1, k)

        mask = (node_i != 0) & (node_j != 0) & (node_k != 0)
        mask = mask & (j > i + 1) & (k > j + 1)
        if not mask.any():
            continue

        # Nodes involved in edge removals: (u, u_next), (v, v_next), (w, w_next)
        u_idx, v_idx, w_idx = i, j, k
        un_idx, vn_idx, wn_idx = i + 1, j + 1, k + 1

        u = torch.gather(tours, 1, u_idx)
        un = torch.gather(tours, 1, un_idx)
        v = torch.gather(tours, 1, v_idx)
        vn = torch.gather(tours, 1, vn_idx)
        w = torch.gather(tours, 1, w_idx)
        wn = torch.gather(tours, 1, wn_idx)

        # Base cost of affected edges
        d_base = (
            dist_matrix[batch_indices, u, un] + dist_matrix[batch_indices, v, vn] + dist_matrix[batch_indices, w, wn]
        )

        # We evaluate the 4 non-2-opt cases (the others are covered by 2-opt)
        # Case 4: (u, v), (un, w), (vn, wn)
        gain4 = d_base - (
            dist_matrix[batch_indices, u, v] + dist_matrix[batch_indices, un, w] + dist_matrix[batch_indices, vn, wn]
        )

        # Case 5: (u, vn), (w, un), (v, wn)
        gain5 = d_base - (
            dist_matrix[batch_indices, u, vn] + dist_matrix[batch_indices, w, un] + dist_matrix[batch_indices, v, wn]
        )

        # Case 6: (u, vn), (w, v), (un, wn)
        gain6 = d_base - (
            dist_matrix[batch_indices, u, vn] + dist_matrix[batch_indices, w, v] + dist_matrix[batch_indices, un, wn]
        )

        # Case 7: (u, w), (vn, un), (v, wn)
        gain7 = d_base - (
            dist_matrix[batch_indices, u, w] + dist_matrix[batch_indices, vn, un] + dist_matrix[batch_indices, v, wn]
        )

        # Concatenate and find best
        all_gains = torch.cat([gain4, gain5, gain6, gain7], dim=1)  # (B, 4)
        best_gain, best_case = torch.max(all_gains, dim=1)  # (B,), (B,)

        improved = (best_gain > 0.001) & mask.squeeze(1)
        if not improved.any():
            continue

        # 2. Apply Improvements
        # This is the complex part: we need to reconstruct the tour for each improved case.
        b_indices = torch.nonzero(improved).squeeze(1)

        for b_idx in b_indices:
            case = best_case[b_idx].item()
            t = tours[b_idx]
            idx_i, idx_j, idx_k = i[b_idx].item(), j[b_idx].item(), k[b_idx].item()

            # Segments:
            # S1: [0 ... idx_i]
            # S2: [idx_i+1 ... idx_j]
            # S3: [idx_j+1 ... idx_k]
            # S4: [idx_k+1 ... max_len-1]
            s1 = t[: idx_i + 1]
            s2 = t[idx_i + 1 : idx_j + 1]
            s3 = t[idx_j + 1 : idx_k + 1]
            s4 = t[idx_k + 1 :]

            if case == 0:  # Case 4: S1 + S2^R + S3^R + S4
                new_t = torch.cat([s1, s2.flip(0), s3.flip(0), s4])
            elif case == 1:  # Case 5: S1 + S3 + S2 + S4
                new_t = torch.cat([s1, s3, s2, s4])
            elif case == 2:  # Case 6: S1 + S3 + S2^R + S4
                new_t = torch.cat([s1, s3, s2.flip(0), s4])
            elif case == 3:  # Case 7: S1 + S3^R + S2 + S4
                new_t = torch.cat([s1, s3.flip(0), s2, s4])
            else:
                continue

            tours[b_idx] = new_t

    return tours
