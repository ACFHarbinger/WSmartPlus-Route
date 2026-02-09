"""
Shaw removal (relatedness-based) operator (vectorized).

The Shaw removal heuristic removes nodes that are similar (related) to a seed node
based on distance, time windows, and demand. This creates clusters of related nodes
that can be efficiently rearranged during repair.
"""

from typing import Optional, Tuple

import torch
from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_shaw_removal(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    n_remove: int,
    demands: Optional[torch.Tensor] = None,
    time_windows: Optional[torch.Tensor] = None,
    phi: float = 9.0,
    chi: float = 3.0,
    psi: float = 2.0,
    randomization_factor: float = 2.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized Shaw removal across a batch of tours using PyTorch.

    Shaw removal computes relatedness between nodes based on multiple criteria:
    - Distance: phi * d(i,j) / max_dist
    - Time windows: chi * |T_i - T_j| / max_time
    - Demand: psi * |q_i - q_j| / max_demand

    Nodes with low relatedness scores (high similarity) to already-removed nodes
    are selected for removal, creating clusters of related nodes.

    Algorithm:
    1. Select random seed node for each tour in batch
    2. While removed < n_remove:
        a. Compute relatedness of all remaining nodes to removed set
        b. Rank by relatedness (lower = more related)
        c. Select using randomized power law: idx = (uniform^p) * len(candidates)
        d. Add to removed set
    3. Return tours with selected nodes marked for removal

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        n_remove: Number of nodes to remove from each tour
        demands: Node demands [B, N+1] or [N+1] (optional, for demand relatedness)
        time_windows: Time windows [B, N+1, 2] or [N+1, 2] (optional, [earliest, latest])
        phi: Weight for distance component (default: 9.0)
        chi: Weight for time window component (default: 3.0)
        psi: Weight for demand component (default: 2.0)
        randomization_factor: Power for randomized selection (default: 2.0)
            Higher values = more randomness

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            - Modified tours [B, N] with removed nodes replaced by padding (-1)
            - Removed nodes [B, n_remove] indices of removed nodes

    Note:
        - Tours should include depot as node 0
        - Removed nodes are marked as -1 in returned tours
        - Works with both batched and shared distance/demand matrices
        - Time windows are optional; if not provided, chi is ignored
    """
    device = distance_matrix.device

    # Handle single tour case
    is_batch = tours.dim() == 2
    if not is_batch:
        tours = tours.unsqueeze(0)

    B, N = tours.shape

    # Expand distance matrix if shared
    if distance_matrix.dim() == 2:
        distance_matrix = distance_matrix.unsqueeze(0)
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Handle demands if provided
    if demands is not None:
        if demands.dim() == 1:
            demands = demands.unsqueeze(0).expand(B, -1)

    # Handle time windows if provided
    if time_windows is not None:
        if time_windows.dim() == 2:
            time_windows = time_windows.unsqueeze(0).expand(B, -1, -1)

    # Compute normalization factors
    max_dist = distance_matrix.max()
    max_demand = demands.max() if demands is not None else torch.tensor(1.0, device=device)
    max_time = (
        (time_windows[:, :, 1] - time_windows[:, :, 0]).max()
        if time_windows is not None
        else torch.tensor(1.0, device=device)
    )

    # Avoid division by zero
    max_dist = torch.clamp(max_dist, min=IMPROVEMENT_EPSILON)
    max_demand = torch.clamp(max_demand, min=IMPROVEMENT_EPSILON)
    max_time = torch.clamp(max_time, min=IMPROVEMENT_EPSILON)

    # Initialize removed nodes tracking
    removed_mask = torch.zeros(B, N, dtype=torch.bool, device=device)
    removed_list = torch.full((B, n_remove), -1, dtype=torch.long, device=device)
    removed_count = torch.zeros(B, dtype=torch.long, device=device)

    # Select random seed node for each batch instance
    valid_mask = tours >= 0  # Non-padding nodes
    valid_counts = valid_mask.sum(dim=1)

    # Random seed selection
    for b in range(B):
        if valid_counts[b] > 0:
            valid_indices = torch.where(valid_mask[b])[0]
            seed_idx = valid_indices[torch.randint(len(valid_indices), (1,), device=device)]
            removed_mask[b, seed_idx] = True
            removed_list[b, 0] = seed_idx
            removed_count[b] = 1

    # Iteratively select related nodes
    for step in range(1, n_remove):
        # For each batch, compute relatedness scores
        relatedness_scores = torch.full((B, N), float("inf"), device=device)

        for b in range(B):
            if removed_count[b] >= n_remove or removed_count[b] >= valid_counts[b]:
                continue

            # Get removed nodes for this batch
            removed_nodes_b = tours[b, removed_mask[b]]
            n_removed = len(removed_nodes_b)

            if n_removed == 0:
                continue

            # Compute relatedness for all non-removed nodes
            for node_idx in range(N):
                if removed_mask[b, node_idx] or not valid_mask[b, node_idx]:
                    continue

                node = tours[b, node_idx]
                total_rel = torch.tensor(0.0, device=device)

                # Average relatedness to all removed nodes
                for rem_node in removed_nodes_b:
                    # Distance component
                    dist_rel = distance_matrix[b, node, rem_node] / max_dist

                    # Demand component
                    demand_rel = torch.tensor(0.0, device=device)
                    if demands is not None:
                        demand_rel = torch.abs(demands[b, node] - demands[b, rem_node]) / max_demand

                    # Time window component
                    time_rel = torch.tensor(0.0, device=device)
                    if time_windows is not None:
                        time_rel = torch.abs(time_windows[b, node, 0] - time_windows[b, rem_node, 0]) / max_time

                    # Combined relatedness
                    rel = phi * dist_rel + chi * time_rel + psi * demand_rel
                    total_rel += rel

                # Average over all removed nodes
                avg_rel = total_rel / n_removed
                relatedness_scores[b, node_idx] = avg_rel

        # Select next node using randomized power law
        for b in range(B):
            if removed_count[b] >= n_remove or removed_count[b] >= valid_counts[b]:
                continue

            # Get candidates (non-removed, non-padding, finite relatedness)
            candidates_mask = (~removed_mask[b]) & valid_mask[b] & (relatedness_scores[b] < float("inf"))

            if not candidates_mask.any():
                break

            candidate_indices = torch.where(candidates_mask)[0]
            candidate_scores = relatedness_scores[b, candidates_mask]

            # Sort by relatedness (lower = more related)
            sorted_indices = torch.argsort(candidate_scores)
            sorted_candidates = candidate_indices[sorted_indices]

            # Randomized selection using power law
            y = torch.rand(1, device=device).item()
            idx = int((y**randomization_factor) * len(sorted_candidates))
            idx = min(idx, len(sorted_candidates) - 1)

            selected_node_idx = sorted_candidates[idx]

            # Mark as removed
            removed_mask[b, selected_node_idx] = True
            removed_list[b, removed_count[b]] = selected_node_idx
            removed_count[b] += 1

    # Create modified tours with removed nodes marked as -1
    modified_tours = tours.clone()
    for b in range(B):
        modified_tours[b, removed_mask[b]] = -1

    return (
        modified_tours if is_batch else modified_tours.squeeze(0),
        removed_list if is_batch else removed_list.squeeze(0),
    )
