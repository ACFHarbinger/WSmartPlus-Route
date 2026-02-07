import torch
from torch import Tensor


def vectorized_greedy_insertion(
    tours: Tensor, removed_nodes: Tensor, dist_matrix: Tensor, demands: Tensor = None, capacity: float = None
) -> Tensor:
    """
    Vectorized greedy insertion.
    Sequentially inserts each node from removed_nodes into the best position in tours.

    Args:
        tours: (B, N_curr)
        removed_nodes: (B, N_rem) - order matters (usually random shuffled before calling)
        dist_matrix: (B, N_all, N_all)
        demands: (N_all) or (B, N_all) - Optional constraint check
        capacity: float - Optional constraint check

    Returns:
        tours: (B, N_curr + N_rem)
    """
    B, N_curr = tours.shape
    B_rem, N_rem = removed_nodes.shape
    device = tours.device

    # Expand dims if needed
    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)

    # We iterate N_rem times
    for i in range(N_rem):
        # Current node to insert: (B, 1)
        node_to_insert = removed_nodes[:, i : i + 1]  # Keep dim

        tours_prev = tours
        tours_next = torch.roll(tours, -1, dims=1)

        # We need cost for each position j
        node_exp = node_to_insert.expand(-1, N_curr)

        batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N_curr)

        d_pn = dist_matrix[batch_idx, tours_prev, node_exp]
        d_nn = dist_matrix[batch_idx, node_exp, tours_next]
        d_pn_existing = dist_matrix[batch_idx, tours_prev, tours_next]

        cost_deltas = d_pn + d_nn - d_pn_existing  # (B, N_curr)

        # Find best position
        best_vals, best_indices = torch.min(cost_deltas, dim=1)  # (B,)

        # Insert node
        N_new = N_curr + 1
        new_tours = torch.zeros((B, N_new), dtype=tours.dtype, device=device)

        seq = torch.arange(N_curr, device=device).unsqueeze(0).expand(B, N_curr)
        insert_pos = best_indices.unsqueeze(1)  # (B, 1) maps to "after index j"

        # Mask for left part: seq <= insert_pos
        mask_left = seq <= insert_pos
        # Mask for right part: seq > insert_pos
        mask_right = ~mask_left

        write_indices = torch.zeros_like(seq)
        write_indices[mask_left] = seq[mask_left]
        write_indices[mask_right] = seq[mask_right] + 1

        new_tours.scatter_(1, write_indices, tours)
        new_tours.scatter_(1, insert_pos + 1, node_to_insert)

        tours = new_tours
        N_curr += 1

    return tours
