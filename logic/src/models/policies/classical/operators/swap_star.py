"""
Swap* local search operator (inter-route swap with re-insertion).
"""

import torch

from logic.src.constants.optimization import IMPROVEMENT_EPSILON


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
        improved = (total_gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            # Apply Swap* moves using a vectorized priority-based indexing
            # We move pos_i to after ins_u and pos_j to after ins_v
            seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len).float()
            weights = seq_b * 10.0

            # For improved instances, change weights of moved indices
            # Node u (at i) moves to after ins_u
            # Node v (at j) moves to after ins_v
            # We use a small tie-breaker if ins_u == ins_v
            weight_u = best_ins_u_idx.float() * 10.0 + 5.0
            weight_v = best_ins_v_idx.float() * 10.0 + 5.1

            # Scatter weights for moved nodes
            # We need to target indices i and j
            weights.scatter_(1, i, weight_u)
            weights.scatter_(1, j, weight_v)

            # Sort weights to get the new index map
            # We only want to apply this to improved rows.
            # To keep it fully vectorized, we always sort and gather, but for non-improved
            # rows we ensure weights stay original.
            orig_weights = seq_b * 10.0
            weights = torch.where(improved, weights, orig_weights)

            idx_map = torch.argsort(weights, dim=1)

            # Apply moves
            tours = torch.gather(tours, 1, idx_map)

    return tours
