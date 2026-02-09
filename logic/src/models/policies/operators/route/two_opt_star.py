"""
2-opt* local search operator (tail swap).
"""

import torch
from logic.src.constants.routing import IMPROVEMENT_EPSILON


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
        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            # Apply tail swap using a vectorized index map
            seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
            idx_map = seq_b.clone()

            # For instances where R1 is before R2 (e_i <= j)
            mask_i_lt_j = (end_i <= j) & improved
            if mask_i_lt_j.any():
                # Define segments
                # s_p1: [0..i] -> stays
                # s_t1: [i+1..e_i-1] -> moves to R2 tail
                # s_gap: [e_i..j] -> shifts
                # s_t2: [j+1..e_j-1] -> moves to R1 tail
                # s_end: [e_j..max_len-1] -> stays

                len_t1 = end_i - (i + 1)
                len_t2 = end_j - (j + 1)
                len_gap = j - end_i + 1

                # New R1 tail starts at i+1
                # Range [i+1, i+len_t2]: maps to [j+1, j+len_t2]
                r1_tail_mask = mask_i_lt_j & (seq_b > i) & (seq_b <= i + len_t2)
                idx_map[r1_tail_mask] = (j + 1).view(B, 1).expand(-1, max_len)[r1_tail_mask] + (
                    seq_b[r1_tail_mask] - (i + 1).view(B, 1).expand(-1, max_len)[r1_tail_mask]
                )

                # Gap shifts
                # Range [i+len_t2+1, i+len_t2+len_gap]: maps to [e_i, j]
                gap_mask = mask_i_lt_j & (seq_b > i + len_t2) & (seq_b <= i + len_t2 + len_gap)
                idx_map[gap_mask] = (end_i).view(B, 1).expand(-1, max_len)[gap_mask] + (
                    seq_b[gap_mask] - (i + len_t2 + 1).view(B, 1).expand(-1, max_len)[gap_mask]
                )

                # New R2 tail starts after gap
                # Range [i+len_t2+len_gap+1, e_j-1]: maps to [i+1, e_i-1]
                r2_tail_mask = mask_i_lt_j & (seq_b > i + len_t2 + len_gap) & (seq_b < end_j.view(B, 1))
                idx_map[r2_tail_mask] = (i + 1).view(B, 1).expand(-1, max_len)[r2_tail_mask] + (
                    seq_b[r2_tail_mask] - (i + len_t2 + len_gap + 1).view(B, 1).expand(-1, max_len)[r2_tail_mask]
                )

            # For instances where R2 is before R1 (e_j <= i)
            mask_j_lt_i = (end_j <= i) & improved
            if mask_j_lt_i.any():
                # Symmetry: same logic but swap i/j and end_i/end_j
                len_t2 = end_j - (j + 1)
                len_t1 = end_i - (i + 1)
                len_gap = i - end_j + 1

                # New R2 tail starts at j+1
                r2_tail_mask = mask_j_lt_i & (seq_b > j) & (seq_b <= j + len_t1)
                idx_map[r2_tail_mask] = (i + 1).view(B, 1).expand(-1, max_len)[r2_tail_mask] + (
                    seq_b[r2_tail_mask] - (j + 1).view(B, 1).expand(-1, max_len)[r2_tail_mask]
                )

                # Gap shifts
                gap_mask = mask_j_lt_i & (seq_b > j + len_t1) & (seq_b <= j + len_t1 + len_gap)
                idx_map[gap_mask] = (end_j).view(B, 1).expand(-1, max_len)[gap_mask] + (
                    seq_b[gap_mask] - (j + len_t1 + 1).view(B, 1).expand(-1, max_len)[gap_mask]
                )

                # New R1 tail starts after gap
                r1_tail_mask = mask_j_lt_i & (seq_b > j + len_t1 + len_gap) & (seq_b < end_i.view(B, 1))
                idx_map[r1_tail_mask] = (j + 1).view(B, 1).expand(-1, max_len)[r1_tail_mask] + (
                    seq_b[r1_tail_mask] - (j + len_t1 + len_gap + 1).view(B, 1).expand(-1, max_len)[r1_tail_mask]
                )

            # Apply tail swap
            tours = torch.gather(tours, 1, idx_map)

    return tours
