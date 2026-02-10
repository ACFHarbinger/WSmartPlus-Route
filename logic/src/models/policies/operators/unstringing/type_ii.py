"""
Type II Unstringing Operator (vectorized).

Removes node V_i and reconnects the route with two reversed segments where k > j.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_ii_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 100,
) -> torch.Tensor:
    """
    Vectorized Type II Unstringing local search across a batch of tours using PyTorch.

    Type II unstringing removes a node V_i and reconnects with two reversed segments:
    - Original: ... V_{i-1} -> V_i -> V_{i+1} ... V_j ... V_k ... (back to i-1)
    - After:    ... V_{i-1} -> V_k <- ... <- V_{j+1}   V_j <- ... <- V_{i+1} ... V_{k+1} ...

    The key difference from Type I is the ordering: here k > j relative to i.

    Reconnection sequence:
        V_{i-1} -> (V_k ... V_{j+1}) reversed -> (V_j ... V_{i+1}) reversed -> V_{k+1} ...

    Algorithm (per batch instance):
    1. For each node position i (candidate for removal):
        a. For sampled positions j, k (where i < j < k in circular order):
            - Compute delta cost of Type II reconnection
            - Delta = -removed_edges + new_edges
        b. Track best (i, j, k) combination
    2. Apply best improving move if found
    3. Repeat until no improvement

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
            Note: Tours should use depot (0) as route separator or be circular
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        max_iterations: Maximum improvement iterations (default: 50)
        sample_size: Number of (j, k) combinations to sample per i (default: 100)
            Set to -1 for exhaustive search.

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours must have at least 7 nodes for meaningful Type II moves
        - Complexity: O(N × sample_size) per iteration
        - Complements Type I by exploring different segment orderings
        - Sample-based approach for computational efficiency

    Mathematical Formulation:
        Given tour T, find (i, j, k) such that j < k (after i):

        Δ = d[v_{i-1}, v_k] + d[v_{j+1}, v_{i+1}] + d[v_j, v_{k+1}]
            - d[v_{i-1}, v_i] - d[v_i, v_{i+1}] - d[v_j, v_{j+1}] - d[v_k, v_{k+1}]

    Example:
        >>> tours = torch.tensor([[0, 5, 3, 8, 2, 7, 1, 0]])
        >>> dist = torch.rand(9, 9)
        >>> improved = vectorized_type_ii_unstringing(tours, dist)
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

    if N < 7:
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

            if len(valid_indices) < 5:
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

                # Sample or enumerate (j, k) pairs
                # Constraint: i < j < k in circular order
                n_valid = len(valid_indices)

                if sample_size > 0:
                    # Sample combinations
                    n_samples = min(sample_size, (n_valid - 2) * (n_valid - 3) // 2)
                    j_samples = torch.randint(i_idx + 1, n_valid - 1, (n_samples,))
                    k_samples = torch.randint(i_idx + 2, n_valid, (n_samples,))

                    # Filter: ensure j < k
                    valid_pairs = k_samples > j_samples
                    j_samples = j_samples[valid_pairs]
                    k_samples = k_samples[valid_pairs]
                else:
                    # Exhaustive search
                    j_list, k_list = [], []
                    for j_idx in range(i_idx + 1, n_valid - 1):
                        for k_idx in range(j_idx + 1, n_valid):
                            j_list.append(j_idx)
                            k_list.append(k_idx)

                    if not j_list:
                        continue

                    j_samples = torch.tensor(j_list, device=device)
                    k_samples = torch.tensor(k_list, device=device)

                if len(j_samples) == 0:
                    continue

                # Evaluate each (j, k) combination
                for j_idx_sample, k_idx_sample in zip(j_samples, k_samples):
                    j = valid_indices[j_idx_sample].item()
                    k = valid_indices[k_idx_sample].item()

                    v_j = tour[j].item()
                    v_k = tour[k].item()

                    j_next = (j + 1) % N
                    k_next = (k + 1) % N

                    v_j_next = tour[j_next].item()
                    v_k_next = tour[k_next].item()

                    # Compute delta cost
                    # Removed edges
                    removed = dist[v_i_prev, v_i] + dist[v_i, v_i_next] + dist[v_j, v_j_next] + dist[v_k, v_k_next]

                    # Inserted edges (Type II pattern)
                    inserted = dist[v_i_prev, v_k] + dist[v_j_next, v_i_next] + dist[v_j, v_k_next]

                    delta = inserted - removed

                    if delta < best_delta - IMPROVEMENT_EPSILON:
                        best_delta = delta
                        best_move = (i, j, k)

            # Apply best move if found
            if best_move is not None:
                i, j, k = best_move

                # Apply Type II unstringing
                tour = _apply_type_ii_move(tour, i, j, k)
                tours[b] = tour
                improved = True

        if not improved:
            break

    return tours if is_batch else tours.squeeze(0)


def _apply_type_ii_move(
    tour: torch.Tensor,
    i: int,
    j: int,
    k: int,
) -> torch.Tensor:
    """
    Apply Type II unstringing move to a tour.

    Removes V_i and reconnects with Type II pattern:
    V_{i-1} -> S2_rev -> S1_rev -> Remainder

    Where:
        S1 = V_{i+1} ... V_j
        S2 = V_{j+1} ... V_k
        Remainder = V_{k+1} ...

    Args:
        tour: Tour tensor [N]
        i: Index of node to remove
        j: Index of V_j
        k: Index of V_k

    Returns:
        torch.Tensor: Modified tour
    """
    n = len(tour)
    tour_list = tour.tolist()

    # Extract segments
    i_next = (i + 1) % n

    # S1: V_{i+1} ... V_j
    s1 = tour_list[i_next : j + 1] if i_next <= j else tour_list[i_next:] + tour_list[: j + 1]

    # S2: V_{j+1} ... V_k
    j_next = (j + 1) % n
    s2 = tour_list[j_next : k + 1] if j_next <= k else tour_list[j_next:] + tour_list[: k + 1]

    # Remainder: V_{k+1} ... V_{i-1}
    k_next = (k + 1) % n
    i_prev = (i - 1) if i > 0 else (n - 1)
    remainder = tour_list[k_next : i_prev + 1] if k_next <= i_prev else tour_list[k_next:] + tour_list[: i_prev + 1]

    # Reconstruct: V_{i-1} -> S2_rev -> S1_rev -> Remainder
    v_i_prev = tour_list[i_prev]
    new_tour = [v_i_prev] + s2[::-1] + s1[::-1] + remainder

    # Restore depot at front if present
    if 0 in new_tour:
        depot_idx = new_tour.index(0)
        new_tour = new_tour[depot_idx:] + new_tour[:depot_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
