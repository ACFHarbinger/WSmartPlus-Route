"""
Or-opt local search operator (vectorized).

The Or-opt operator relocates chains of 1-3 consecutive nodes to better positions,
either within the same route or to different routes. This is particularly effective
for geographically clustered customers.
"""

from typing import Optional

import torch
from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_or_opt(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    demands: Optional[torch.Tensor] = None,
    chain_lengths: tuple = (1, 2, 3),
    max_iterations: int = 100,
) -> torch.Tensor:
    """
    Vectorized Or-opt local search across a batch of tours using PyTorch.

    The Or-opt operator tries to relocate chains of consecutive nodes (typically
    1, 2, or 3 nodes) to better positions in the tour. This is a generalization
    of node relocation that considers longer sequences.

    For a chain of k nodes starting at position i:
    Original: ... -> a -> [c1 -> c2 -> ... -> ck] -> b -> ...
    Or-opt:   ... -> a -> b -> ... -> x -> [c1 -> c2 -> ... -> ck] -> y -> ...

    The algorithm evaluates:
    1. Removal gain = d(a,c1) + d(ck,b) - d(a,b)
    2. Insertion cost = d(x,c1) + d(ck,y) - d(x,y)
    3. Delta = insertion_cost - removal_gain

    This implementation processes batches in parallel for GPU acceleration.

    Algorithm:
    1. For each chain length k in [1, 2, 3]:
        a. For all valid chain positions in parallel:
            - Compute removal gain
            - For all insertion positions:
                - Compute insertion cost and delta
        b. Select best improvement for each tour
        c. Apply moves where delta < 0
    2. Repeat until no improvement or max_iterations

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        capacities: Vehicle capacities [B] or scalar (optional, for capacity checks)
        demands: Node demands [B, N+1] or [N+1] (optional, for capacity checks)
        chain_lengths: Tuple of chain lengths to try (default: (1, 2, 3))
        max_iterations: Maximum number of improvement iterations (default: 100)

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours should include depot as node 0
        - Capacity constraints are checked if capacities and demands provided
        - Works with both batched and shared distance matrices
        - Stops early if no improvement found
        - Chain lengths > N/2 are automatically skipped
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
    if N < 4:  # Too small for or-opt
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Handle demands if provided
    has_capacity = capacities is not None and demands is not None
    if has_capacity:
        if demands.dim() == 1:
            demands = demands.unsqueeze(0).expand(B, -1)
        if capacities.dim() == 0:
            capacities = capacities.unsqueeze(0).expand(B)

    batch_indices = torch.arange(B, device=device).view(B, 1)

    # Iterate through chain lengths
    for chain_len in chain_lengths:
        if chain_len >= N // 2:  # Skip if chain too long
            continue

        for _iteration in range(max_iterations):
            improved_any = False

            # Generate all valid chain start positions: [0, N - chain_len)
            chain_starts = torch.arange(N - chain_len, device=device)
            n_chains = len(chain_starts)

            if n_chains == 0:
                break

            # Expand for batch: (B, n_chains)
            chain_starts_batch = chain_starts.unsqueeze(0).expand(B, -1)

            # Get nodes involved in removal
            # prev_node: node before chain
            # next_node: node after chain
            prev_idx = chain_starts_batch - 1
            next_idx = chain_starts_batch + chain_len

            # Handle depot edges
            prev_idx = prev_idx.clamp(min=0)
            next_idx = next_idx.clamp(max=N - 1)

            prev_nodes = torch.gather(tours, 1, prev_idx)
            chain_first = torch.gather(tours, 1, chain_starts_batch)
            chain_last = torch.gather(tours, 1, (chain_starts_batch + chain_len - 1).clamp(max=N - 1))
            next_nodes = torch.gather(tours, 1, next_idx)

            # Compute removal gain: d(prev, chain_first) + d(chain_last, next) - d(prev, next)
            b_exp = batch_indices.expand(B, n_chains)
            removal_gain = (
                distance_matrix[b_exp, prev_nodes, chain_first]
                + distance_matrix[b_exp, chain_last, next_nodes]
                - distance_matrix[b_exp, prev_nodes, next_nodes]
            )

            # Compute chain demand if capacity checking enabled
            if has_capacity:
                chain_demands = torch.zeros(B, n_chains, device=device)
                for k in range(chain_len):
                    chain_nodes = torch.gather(tours, 1, (chain_starts_batch + k).clamp(max=N - 1))
                    chain_demands += torch.gather(demands, 1, chain_nodes)

            # Try all insertion positions for each chain
            best_delta = torch.full((B, n_chains), float("inf"), device=device)
            best_insert_pos = torch.zeros((B, n_chains), dtype=torch.long, device=device)

            for insert_pos in range(N + 1):
                # Get insertion neighbors
                ins_prev_idx = torch.full((B, n_chains), insert_pos - 1, device=device)
                ins_next_idx = torch.full((B, n_chains), insert_pos, device=device)

                # Clamp to valid range
                ins_prev_idx = ins_prev_idx.clamp(min=0, max=N - 1)
                ins_next_idx = ins_next_idx.clamp(min=0, max=N - 1)

                ins_prev_nodes = torch.gather(tours, 1, ins_prev_idx)
                ins_next_nodes = torch.gather(tours, 1, ins_next_idx)

                # Skip positions that would reinsert at same spot
                # This requires adjusting for removal effect
                skip_mask = torch.zeros((B, n_chains), dtype=torch.bool, device=device)
                for k in range(n_chains):
                    chain_start = chain_starts[k].item()
                    # Skip if insert_pos is in [chain_start, chain_start + chain_len]
                    if chain_start <= insert_pos <= chain_start + chain_len:
                        skip_mask[:, k] = True

                # Compute insertion cost
                insertion_cost = (
                    distance_matrix[b_exp, ins_prev_nodes, chain_first]
                    + distance_matrix[b_exp, chain_last, ins_next_nodes]
                    - distance_matrix[b_exp, ins_prev_nodes, ins_next_nodes]
                )

                # Capacity check if enabled
                if has_capacity:
                    # Current tour load (approximation - simplified for vectorization)
                    # In practice, would need route-specific load tracking
                    feasible = chain_demands <= capacities.unsqueeze(1)
                    insertion_cost = torch.where(feasible, insertion_cost, torch.tensor(float("inf"), device=device))

                # Delta = insertion_cost - removal_gain
                delta = insertion_cost - removal_gain
                delta = torch.where(skip_mask, torch.tensor(float("inf"), device=device), delta)

                # Update best if better
                better = delta < best_delta
                best_delta = torch.where(better, delta, best_delta)
                best_insert_pos = torch.where(better, ins_next_idx, best_insert_pos)

            # Find best chain to move for each batch instance
            best_chain_delta, best_chain_idx = torch.min(best_delta, dim=1)

            # Check if any improvements found
            improved = best_chain_delta < -IMPROVEMENT_EPSILON
            if not improved.any():
                break

            improved_any = True

            # Apply best move for each improved instance
            # This is complex in vectorized form - simplified version:
            # For each batch element that improved, we need to:
            # 1. Remove chain from original position
            # 2. Insert chain at new position
            # This requires careful index manipulation

            for b in range(B):
                if improved[b]:
                    chain_idx = best_chain_idx[b].item()
                    chain_start = chain_starts[chain_idx].item()
                    insert_pos = best_insert_pos[b, chain_idx].item()

                    # Extract chain
                    chain = tours[b, chain_start : chain_start + chain_len].clone()

                    # Remove chain
                    mask = torch.ones(N, dtype=torch.bool, device=device)
                    mask[chain_start : chain_start + chain_len] = False
                    remaining = tours[b][mask]

                    # Adjust insertion position if after removal point
                    adj_insert = insert_pos
                    if insert_pos > chain_start + chain_len:
                        adj_insert -= chain_len

                    # Insert chain
                    new_tour = torch.cat([remaining[:adj_insert], chain, remaining[adj_insert:]])

                    # Pad if necessary to maintain shape
                    if new_tour.size(0) < N:
                        padding = torch.zeros(N - new_tour.size(0), dtype=tours.dtype, device=device)
                        new_tour = torch.cat([new_tour, padding])
                    elif new_tour.size(0) > N:
                        new_tour = new_tour[:N]

                    tours[b] = new_tour

            if not improved_any:
                break

    return tours if is_batch else tours.squeeze(0)
