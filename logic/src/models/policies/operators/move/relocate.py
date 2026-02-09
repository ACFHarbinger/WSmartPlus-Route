"""
Relocate local search operator.
"""

import torch
from logic.src.constants.routing import IMPROVEMENT_EPSILON


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

        improved = (gain > IMPROVEMENT_EPSILON) & mask

        improved = (gain > IMPROVEMENT_EPSILON) & mask

        if improved.any():
            # Apply relocate using a vectorized index map
            seq_b = torch.arange(max_len, device=device).view(1, max_len).expand(B, max_len)
            idx_map = seq_b.clone()

            # For each instance, we want to move i to after j
            # Case 1: i < j. Sequence: [0..i-1], [i+1..j], i, [j+1..N-1]
            mask_i_lt_j = (i < j) & improved
            # Elements in [i, j): index shifts left
            shift_left = mask_i_lt_j & (seq_b >= i) & (seq_b < j)
            idx_map[shift_left] = seq_b[shift_left] + 1
            # Element at j mapping to original i
            target_mask = mask_i_lt_j & (seq_b == j)
            idx_map[target_mask] = i.expand(-1, max_len)[target_mask]

            # Case 2: j < i. Sequence: [0..j], i, [j+1..i-1], [i+1..N-1]
            mask_j_lt_i = (j < i) & improved
            # Elements in (j, i): index shifts right
            shift_right = mask_j_lt_i & (seq_b > j) & (seq_b < i)
            idx_map[shift_right] = seq_b[shift_right] - 1
            # Element at j+1 mapping to original i
            target_mask = mask_j_lt_i & (seq_b == j + 1)
            idx_map[target_mask] = i.expand(-1, max_len)[target_mask]

            # Apply relocation
            tours = torch.gather(tours, 1, idx_map)

    return tours
