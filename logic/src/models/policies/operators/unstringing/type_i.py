"""
Type I Unstringing Operator (vectorized).

Removes node V_i and reconnects the route by reversing two sub-tours.
This operator deletes 4 arcs and inserts 3 arcs with segment reversals.
"""

import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_i_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 100,
) -> torch.Tensor:
    """
    Vectorized Type I Unstringing local search across a batch of tours using PyTorch.

    Type I unstringing removes a node V_i and reconnects by reversing two sub-tours:
    - Original: ... V_{i-1} -> V_i -> V_{i+1} ... V_k -> V_{k+1} ... V_j -> V_{j+1} ...
    - After:    ... V_{i-1} -> V_k <- ... <- V_{i+1}   V_j <- ... <- V_{k+1}   V_{j+1} ...

    Deleted arcs:
        (V_{i-1}, V_i), (V_i, V_{i+1}), (V_k, V_{k+1}), (V_j, V_{j+1})

    Inserted arcs:
        (V_{i-1}, V_k), (V_{i+1}, V_j), (V_{k+1}, V_{j+1})

    This creates a 4-opt style move with two segment reversals.

    Algorithm (per batch instance):
    1. For each node position i (candidate for removal):
        a. For sampled positions j, k (where i < k < j in circular order):
            - Compute delta cost of removing V_i and reversing segments
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
            Higher = more thorough but slower. Set to -1 for exhaustive search.

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours must have at least 7 nodes for meaningful Type I moves
        - Complexity: O(N × sample_size) per iteration
        - Sample-based approach trades optimality for speed
        - For small tours (N < 20), set sample_size=-1 for exhaustive search
        - This is a pure improvement operator (does not guarantee feasibility)

    Mathematical Formulation:
        Given tour T = (v_0, ..., v_n), find (i, j, k) such that:

        Δ = d[v_{i-1}, v_k] + d[v_{i+1}, v_j] + d[v_{k+1}, v_{j+1}]
            - d[v_{i-1}, v_i] - d[v_i, v_{i+1}] - d[v_k, v_{k+1}] - d[v_j, v_{j+1}]

        Subject to: 0 < i < k < j < n (circular indices)

    Example:
        >>> tours = torch.tensor([[0, 5, 3, 8, 2, 7, 1, 0]])
        >>> dist = torch.rand(9, 9)
        >>> improved = vectorized_type_i_unstringing(tours, dist)
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

    if N < 7:  # Need at least 7 nodes for meaningful Type I moves
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

            # Find non-depot nodes (assuming depot is 0)
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
                # Constraint: i < k < j in circular order
                n_valid = len(valid_indices)

                if sample_size > 0:
                    # Sample combinations
                    n_samples = min(sample_size, (n_valid - 2) * (n_valid - 3) // 2)
                    k_samples = torch.randint(i_idx + 2, n_valid - 1, (n_samples,))
                    j_samples = torch.randint(0, n_valid, (n_samples,))

                    # Filter: ensure k < j in circular order
                    valid_pairs = j_samples > k_samples
                    k_samples = k_samples[valid_pairs]
                    j_samples = j_samples[valid_pairs]
                else:
                    # Exhaustive search
                    k_list, j_list = [], []
                    for k_idx in range(i_idx + 2, n_valid - 1):
                        for j_idx in range(k_idx + 1, n_valid):
                            k_list.append(k_idx)
                            j_list.append(j_idx)

                    if not k_list:
                        continue

                    k_samples = torch.tensor(k_list, device=device)
                    j_samples = torch.tensor(j_list, device=device)

                if len(k_samples) == 0:
                    continue

                # Evaluate each (j, k) combination
                for k_idx_sample, j_idx_sample in zip(k_samples, j_samples):
                    k = valid_indices[k_idx_sample].item()
                    j = valid_indices[j_idx_sample].item()

                    v_k = tour[k].item()
                    v_j = tour[j].item()

                    k_next = (k + 1) % N
                    j_next = (j + 1) % N

                    v_k_next = tour[k_next].item()
                    v_j_next = tour[j_next].item()

                    # Compute delta cost
                    # Removed edges
                    removed = dist[v_i_prev, v_i] + dist[v_i, v_i_next] + dist[v_k, v_k_next] + dist[v_j, v_j_next]

                    # Inserted edges
                    inserted = dist[v_i_prev, v_k] + dist[v_i_next, v_j] + dist[v_k_next, v_j_next]

                    delta = inserted - removed

                    if delta < best_delta - IMPROVEMENT_EPSILON:
                        best_delta = delta
                        best_move = (i, j, k)

            # Apply best move if found
            if best_move is not None:
                i, j, k = best_move

                # Apply Type I unstringing
                tour = _apply_type_i_move(tour, i, j, k)
                tours[b] = tour
                improved = True

        if not improved:
            break

    return tours if is_batch else tours.squeeze(0)


def _apply_type_i_move(
    tour: torch.Tensor,
    i: int,
    j: int,
    k: int,
) -> torch.Tensor:
    """
    Apply Type I unstringing move to a tour.

    Removes V_i and reconnects with reversed segments.

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
    # Segment 1: V_{i+1} ... V_k
    i_next = (i + 1) % n
    seg1 = tour_list[i_next : k + 1] if i_next <= k else tour_list[i_next:] + tour_list[: k + 1]

    # Segment 2: V_{k+1} ... V_j
    k_next = (k + 1) % n
    seg2 = tour_list[k_next : j + 1] if k_next <= j else tour_list[k_next:] + tour_list[: j + 1]

    # Remainder: V_{j+1} ... V_{i-1}
    j_next = (j + 1) % n
    i_prev = (i - 1) if i > 0 else (n - 1)
    remainder = tour_list[j_next : i_prev + 1] if j_next <= i_prev else tour_list[j_next:] + tour_list[: i_prev + 1]

    # Reconstruct: V_{i-1} -> seg1_rev -> seg2_rev -> remainder
    v_i_prev = tour_list[i_prev]
    new_tour = [v_i_prev] + seg1[::-1] + seg2[::-1] + remainder

    # Restore depot at front if present
    if 0 in new_tour:
        depot_idx = new_tour.index(0)
        new_tour = new_tour[depot_idx:] + new_tour[:depot_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
