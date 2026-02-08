from typing import Tuple

import torch
from torch import Tensor


def vectorized_worst_removal(tours: Tensor, dist_matrix: Tensor, n_remove: int) -> Tuple[Tensor, Tensor]:
    """
    Vectorized worst removal (highest saving).

    Args:
        tours (Tensor): (B, N)
        dist_matrix (Tensor): (B, N_all, N_all) or (1, N_all, N_all)
        n_remove (int): Number to remove

    Returns:
        Tuple[Tensor, Tensor]: New tours, Removed nodes
    """
    B, N = tours.shape
    device = tours.device

    if dist_matrix.dim() == 2:
        dist_matrix = dist_matrix.unsqueeze(0).expand(B, -1, -1)
    elif dist_matrix.size(0) == 1 and B > 1:
        dist_matrix = dist_matrix.expand(B, -1, -1)

    # Calculate savings for each node
    # prev -> node -> next
    # Cost = d(prev, node) + d(node, next)
    # New Cost = d(prev, next)
    # Saving = Cost - New Cost

    # Shifted tours
    tours_prev = torch.roll(tours, 1, dims=1)
    tours_next = torch.roll(tours, -1, dims=1)

    # Helper to gather distances
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(B, N)

    d_prev_node = dist_matrix[batch_idx, tours_prev, tours]
    d_node_next = dist_matrix[batch_idx, tours, tours_next]
    d_prev_next = dist_matrix[batch_idx, tours_prev, tours_next]

    savings = d_prev_node + d_node_next - d_prev_next

    # Mask out depots/padding (should not remove them)
    customers_mask = tours > 0
    savings[~customers_mask] = -float("inf")

    # Pick top n_remove savings
    _, remove_indices = torch.topk(savings, k=n_remove, dim=1)

    removed_nodes = torch.gather(tours, 1, remove_indices)

    # Create mask and collapse
    remove_mask = torch.zeros((B, N), dtype=torch.bool, device=device)
    remove_mask.scatter_(1, remove_indices, True)

    keep_mask = ~remove_mask
    new_tours = tours[keep_mask].view(B, N - n_remove)

    return new_tours, removed_nodes
