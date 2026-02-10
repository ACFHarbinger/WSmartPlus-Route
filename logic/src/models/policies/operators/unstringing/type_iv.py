"""
Type IV Unstringing Operator (vectorized).

Most complex US operator involving four neighbor nodes and multiple reversals.
Provides the deepest exploration of the unstringing neighborhood.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_iv_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 30,
) -> torch.Tensor:
    """
    Vectorized Type IV Unstringing local search across a batch of tours using PyTorch.

    Type IV is the most sophisticated unstringing operator, involving four positions
    (i, j, k, l) with complex segment rearrangement and selective reversals.

    Removes V_i. Involves neighbors V_j, V_l, V_k such that in rotated order:
        V_{i+1} ... V_j ... V_l ... V_k ... V_{i-1}

    Deletes arcs:
        (V_{j-1}, V_j), (V_{l-1}, V_l), (V_k, V_{k+1}), (V_{i-1}, V_i), (V_i, V_{i+1})

    Inserts arcs:
        (V_{j-1}, V_l), (V_k, V_{i-1}), (V_{k+1}, V_{l-1}), (V_j, V_{i+1})

    Reconstructs as:
        S_C + S_D + S_A_rev + S_B_rev

    Where:
        S_C = (V_{i+1} ... V_{j-1})
        S_D = (V_l ... V_k)
        S_A = (V_{k+1} ... V_{i-1}) -> reversed
        S_B = (V_j ... V_{l-1}) -> reversed

    This creates a complex 5-opt style move with selective reversals, making it the
    most powerful operator for escaping deep local optima.

    Algorithm (per batch instance):
    1. For each node position i (candidate for removal):
        a. For sampled positions j, l, k (where i < j < l < k in circular order):
            - Compute delta cost of Type IV reconnection
            - Delta = -removed_edges + new_edges
        b. Track best (i, j, l, k) combination
    2. Apply best improving move if found
    3. Repeat until no improvement

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
            Note: Tours should use depot (0) as route separator or be circular
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        max_iterations: Maximum improvement iterations (default: 50)
        sample_size: Number of (j, l, k) combinations to sample per i (default: 30)
            Lower than Type III due to O(N³) complexity. Set to -1 for exhaustive search.

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours must have at least 10 nodes for meaningful Type IV moves
        - Complexity: O(N × sample_size) per iteration, but sample_size covers O(N³) space
        - Most computationally expensive but also most powerful unstringing operator
        - Significantly reduced default sample_size due to complexity
        - Best used as final refinement step after simpler operators

    Mathematical Formulation:
        Given tour T, find (i, j, l, k) such that i < j < l < k (circular):

        Δ = d[v_{j-1}, v_l] + d[v_k, v_{i-1}] + d[v_{k+1}, v_{l-1}] + d[v_j, v_{i+1}]
            - d[v_{j-1}, v_j] - d[v_{l-1}, v_l]
            - d[v_k, v_{k+1}] - d[v_{i-1}, v_i] - d[v_i, v_{i+1}]

    Example:
        >>> tours = torch.tensor([[0, 5, 3, 8, 2, 7, 1, 4, 6, 9, 0]])
        >>> dist = torch.rand(11, 11)
        >>> improved = vectorized_type_iv_unstringing(tours, dist)
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

    if N < 10:  # Need at least 10 nodes for Type IV
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Main improvement loop
    for _iteration in range(max_iterations):
        improved = False

        # Process each batch instance
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            # Find valid nodes
            valid_nodes = (tour > 0) & (tour < N)
            valid_indices = torch.where(valid_nodes)[0]

            if len(valid_indices) < 8:
                continue

            best_delta = 0.0
            best_move = None

            # Try removing each node i
            for i_idx in range(len(valid_indices)):
                i = valid_indices[i_idx].item()

                # Get neighbors
                i_prev = (i - 1) if i > 0 else (N - 1)
                i_next = (i + 1) % N

                v_i_prev = tour[i_prev].item()
                v_i = tour[i].item()
                v_i_next = tour[i_next].item()

                # Sample or enumerate (j, l, k) triples
                # Constraint: i < j < l < k in circular order
                n_valid = len(valid_indices)

                if sample_size > 0:
                    # Sample combinations
                    n_samples = min(
                        sample_size,
                        (n_valid - 3) * (n_valid - 4) * (n_valid - 5) // 6,
                    )

                    # Random sampling with sorting to ensure j < l < k
                    samples = torch.randint(i_idx + 1, n_valid, (n_samples, 3), device=device)
                    samples, _ = torch.sort(samples, dim=1)
                    j_samples = samples[:, 0]
                    l_samples = samples[:, 1]
                    k_samples = samples[:, 2]

                    # Filter duplicates
                    valid_mask = (j_samples < l_samples) & (l_samples < k_samples)
                    j_samples = j_samples[valid_mask]
                    l_samples = l_samples[valid_mask]
                    k_samples = k_samples[valid_mask]
                else:
                    # Exhaustive search
                    j_list, l_list, k_list = [], [], []
                    for j_idx in range(i_idx + 1, n_valid - 2):
                        for l_idx in range(j_idx + 1, n_valid - 1):
                            for k_idx in range(l_idx + 1, n_valid):
                                j_list.append(j_idx)
                                l_list.append(l_idx)
                                k_list.append(k_idx)

                    if not j_list:
                        continue

                    j_samples = torch.tensor(j_list, device=device)
                    l_samples = torch.tensor(l_list, device=device)
                    k_samples = torch.tensor(k_list, device=device)

                if len(j_samples) == 0:
                    continue

                # Evaluate each (j, l, k) combination
                for j_idx_sample, l_idx_sample, k_idx_sample in zip(j_samples, l_samples, k_samples):
                    j = valid_indices[j_idx_sample].item()
                    l = valid_indices[l_idx_sample].item()
                    k = valid_indices[k_idx_sample].item()

                    v_j = tour[j].item()
                    v_l = tour[l].item()
                    v_k = tour[k].item()

                    j_prev = (j - 1) if j > 0 else (N - 1)
                    l_prev = (l - 1) if l > 0 else (N - 1)
                    k_next = (k + 1) % N

                    v_j_prev = tour[j_prev].item()
                    v_l_prev = tour[l_prev].item()
                    v_k_next = tour[k_next].item()

                    # Compute delta cost
                    # Removed edges (Type IV pattern)
                    removed = (
                        dist[v_j_prev, v_j]
                        + dist[v_l_prev, v_l]
                        + dist[v_k, v_k_next]
                        + dist[v_i_prev, v_i]
                        + dist[v_i, v_i_next]
                    )

                    # Inserted edges (Type IV pattern)
                    inserted = (
                        dist[v_j_prev, v_l] + dist[v_k, v_i_prev] + dist[v_k_next, v_l_prev] + dist[v_j, v_i_next]
                    )

                    delta = inserted - removed

                    if delta < best_delta - IMPROVEMENT_EPSILON:
                        best_delta = delta
                        best_move = (i, j, l, k)

            # Apply best move if found
            if best_move is not None:
                i, j, l, k = best_move

                # Apply Type IV unstringing
                tour = _apply_type_iv_move(tour, i, j, l, k)
                tours[b] = tour
                improved = True

        if not improved:
            break

    return tours if is_batch else tours.squeeze(0)


def _apply_type_iv_move(
    tour: torch.Tensor,
    i: int,
    j: int,
    l: int,
    k: int,
) -> torch.Tensor:
    """
    Apply Type IV unstringing move to a tour.

    Removes V_i and reconnects with Type IV pattern:
    S_C + S_D + S_A_rev + S_B_rev

    Where:
        S_C = V_{i+1} ... V_{j-1}
        S_D = V_l ... V_k
        S_A = V_{k+1} ... V_{i-1} (reversed)
        S_B = V_j ... V_{l-1} (reversed)

    Args:
        tour: Tour tensor [N]
        i: Index of node to remove
        j: Index of V_j
        l: Index of V_l
        k: Index of V_k

    Returns:
        torch.Tensor: Modified tour
    """
    n = len(tour)
    tour_list = tour.tolist()

    # Note: V_i is removed, so we work with segments around it
    i_next = (i + 1) % n
    i_prev = (i - 1) if i > 0 else (n - 1)

    # S_C: V_{i+1} ... V_{j-1}
    j_prev = (j - 1) if j > 0 else (n - 1)
    s_c = tour_list[i_next : j_prev + 1] if i_next <= j_prev else tour_list[i_next:] + tour_list[: j_prev + 1]

    # S_B: V_j ... V_{l-1}
    l_prev = (l - 1) if l > 0 else (n - 1)
    s_b = tour_list[j : l_prev + 1] if j <= l_prev else tour_list[j:] + tour_list[: l_prev + 1]

    # S_D: V_l ... V_k
    s_d = tour_list[l : k + 1] if l <= k else tour_list[l:] + tour_list[: k + 1]

    # S_A: V_{k+1} ... V_{i-1}
    k_next = (k + 1) % n
    s_a = tour_list[k_next : i_prev + 1] if k_next <= i_prev else tour_list[k_next:] + tour_list[: i_prev + 1]

    # Reconstruct: S_C + S_D + S_A_rev + S_B_rev
    new_tour = s_c + s_d + s_a[::-1] + s_b[::-1]

    # Restore depot at front if present
    if 0 in new_tour:
        depot_idx = new_tour.index(0)
        new_tour = new_tour[depot_idx:] + new_tour[:depot_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
