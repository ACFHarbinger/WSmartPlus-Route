"""
3-opt local search operator.
"""

import torch
from logic.src.constants.optimization import IMPROVEMENT_EPSILON


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

        improved = (best_gain > IMPROVEMENT_EPSILON) & mask.squeeze(1)
        if improved.any():
            # Apply 3-opt improvements using vectorized index maps
            seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
            idx_map = seq_b.clone()

            # For each case, we define how indices map
            # We want to select based on best_case
            # Indices for segments
            # S1: [0..i], S2: [i+1..j], S3: [j+1..k], S4: [k+1..max_len-1]
            len3 = k - j

            # Construct masks for segments in a hypothetical newly ordered tour
            # All cases keep S1 and S4 in place.
            # Case 4 (idx 0): S1, S2^R, S3^R, S4
            mask0 = (best_case == 0) & improved
            # S2^R: [i+1, j] maps to original [i+1, j] reversed
            c0_s2 = mask0.view(B, 1) & (seq_b > i) & (seq_b <= j)
            idx_map[c0_s2] = i.expand(-1, max_len)[c0_s2] + j.expand(-1, max_len)[c0_s2] + 1 - seq_b[c0_s2]
            # S3^R: [j+1, k] maps to original [j+1, k] reversed
            c0_s3 = mask0.view(B, 1) & (seq_b > j) & (seq_b <= k)
            idx_map[c0_s3] = j.expand(-1, max_len)[c0_s3] + k.expand(-1, max_len)[c0_s3] + 1 - seq_b[c0_s3]

            # Case 5 (idx 1): S1, S3, S2, S4
            mask1 = (best_case == 1) & improved
            # S3: [j+1..k] moves to [i+1..i+len3]
            c1_s3 = mask1.view(B, 1) & (seq_b > i) & (seq_b <= i + len3)
            # mapping: new_idx = orig_idx + shift. shift = (j+1) - (i+1) = j-i
            # new k_idx = orig j+1. k_idx = i+1. (j+1) - (i+1) = j-i. checks out.
            idx_map[c1_s3] = seq_b[c1_s3] + (j - i).view(B, 1).expand(-1, max_len)[c1_s3]
            # S2: [i+1..j] moves to [i+len3+1..k]
            c1_s2 = mask1.view(B, 1) & (seq_b > i + len3) & (seq_b <= k)
            # mapping: shift = (i+1) - (j+1) = i-j == -(j-i)
            # range len = j-i.
            # new range start = i+len3+1 = i + (k-j) + 1.
            # orig range start = i+1.
            # wait, S2 has len (j-i). S3 has len (k-j).
            # new pos of S2 starts after S3.
            # new loc = i + len(S3) + 1 + offset.
            # mapping back to old: old = new - (k-j)
            idx_map[c1_s2] = seq_b[c1_s2] - (k - j).view(B, 1).expand(-1, max_len)[c1_s2]

            # Case 6 (idx 2): S1, S3, S2^R, S4
            mask2 = (best_case == 2) & improved
            # S3: [j+1..k] moves to [i+1..i+len3] -> shift + (j-i)
            c2_s3 = mask2.view(B, 1) & (seq_b > i) & (seq_b <= i + len3)
            idx_map[c2_s3] = seq_b[c2_s3] + (j - i).view(B, 1).expand(-1, max_len)[c2_s3]
            # S2^R: [i+1..j] -> reversed -> moves to [i+len3+1..k]
            # Target range [T_start, T_end]. Source [S_start, S_end].
            # Map T_k -> S_end - (T_k - T_start).
            c2_s2 = mask2.view(B, 1) & (seq_b > i + len3) & (seq_b <= k)
            idx_map[c2_s2] = j.expand(-1, max_len)[c2_s2] - (
                seq_b[c2_s2] - (i + len3 + 1).view(B, 1).expand(-1, max_len)[c2_s2]
            )

            # Case 7 (idx 3): S1, S3^R, S2, S4
            mask3 = (best_case == 3) & improved
            # S3^R: [j+1..k] reversed -> moves to [i+1..i+len3]
            # Target [i+1..i+len3]. Source [j+1..k].
            c3_s3 = mask3.view(B, 1) & (seq_b > i) & (seq_b <= i + len3)
            idx_map[c3_s3] = k.expand(-1, max_len)[c3_s3] - (
                seq_b[c3_s3] - (i + 1).view(B, 1).expand(-1, max_len)[c3_s3]
            )
            # S2: [i+1..j] moves to [i+len3+1..k]
            # simple shift right by (k-j)
            c3_s2 = mask3.view(B, 1) & (seq_b > i + len3) & (seq_b <= k)
            idx_map[c3_s2] = seq_b[c3_s2] - (k - j).view(B, 1).expand(-1, max_len)[c3_s2]

            # Apply
            tours = torch.gather(tours, 1, idx_map)

    return tours
