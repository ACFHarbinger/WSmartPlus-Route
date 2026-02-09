"""
2-opt local search operator.
"""

import torch
from logic.src.constants.routing import IMPROVEMENT_EPSILON


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
        improved = best_gain > IMPROVEMENT_EPSILON
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
