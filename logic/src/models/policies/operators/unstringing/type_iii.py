"""
Type III Unstringing Operator (vectorized).

Involves three neighbor nodes (V_j, V_k, V_l) and complex triple reversals.
Most sophisticated unstringing variant with three segment reversals.
"""


import torch

from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_type_iii_unstringing(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    max_iterations: int = 50,
    sample_size: int = 50,
) -> torch.Tensor:
    """
    Vectorized Type III Unstringing local search across a batch of tours using PyTorch.

    Type III is the most complex unstringing operator, involving three neighbor nodes
    and triple segment reversals. It removes V_i and reconnects using neighbors V_j, V_k, V_l.

    Order in route (relative to i): V_{i+1} ... V_k ... V_j ... V_l ... V_{i-1}

    Deleted arcs:
        (V_{i-1}, V_i), (V_i, V_{i+1}), (V_k, V_{k+1}), (V_j, V_{j+1}), (V_l, V_{l+1})

    Inserted arcs:
        (V_{i-1}, V_k), (V_{i+1}, V_j), (V_{k+1}, V_l), (V_{j+1}, V_{l+1})

    Reconstructs as:
        V_{i-1} -> S1_rev -> S2_rev -> S3_rev -> Remainder

    Where:
        S1 = (V_{i+1} ... V_k)
        S2 = (V_{k+1} ... V_j)
        S3 = (V_{j+1} ... V_l)
        Remainder = (V_{l+1} ... back to start)

    This creates a 5-opt style move with three segment reversals, making it extremely
    powerful for escaping local optima.

    Algorithm (per batch instance):
    1. For each node position i (candidate for removal):
        a. For sampled positions k, j, l (where i < k < j < l in circular order):
            - Compute delta cost of Type III reconnection
            - Delta = -removed_edges + new_edges
        b. Track best (i, k, j, l) combination
    2. Apply best improving move if found
    3. Repeat until no improvement

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
            Note: Tours should use depot (0) as route separator or be circular
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        max_iterations: Maximum improvement iterations (default: 50)
        sample_size: Number of (k, j, l) combinations to sample per i (default: 50)
            Lower than Type I/II due to O(N³) complexity. Set to -1 for exhaustive search.

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours must have at least 9 nodes for meaningful Type III moves
        - Complexity: O(N × sample_size) per iteration, but sample_size covers O(N³) space
        - Most powerful unstringing operator for escaping local optima
        - Reduced default sample_size due to computational cost
        - For tours with N > 30, sampling is strongly recommended

    Mathematical Formulation:
        Given tour T, find (i, k, j, l) such that i < k < j < l (circular):

        Δ = d[v_{i-1}, v_k] + d[v_{i+1}, v_j] + d[v_{k+1}, v_l] + d[v_{j+1}, v_{l+1}]
            - d[v_{i-1}, v_i] - d[v_i, v_{i+1}]
            - d[v_k, v_{k+1}] - d[v_j, v_{j+1}] - d[v_l, v_{l+1}]

    Example:
        >>> tours = torch.tensor([[0, 5, 3, 8, 2, 7, 1, 4, 6, 0]])
        >>> dist = torch.rand(10, 10)
        >>> improved = vectorized_type_iii_unstringing(tours, dist)
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

    if N < 9:  # Need at least 9 nodes for Type III
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Main improvement loop
    for iteration in range(max_iterations):
        improved = False

        # Process each batch instance
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            # Find valid nodes
            valid_nodes = (tour > 0) & (tour < N)
            valid_indices = torch.where(valid_nodes)[0]

            if len(valid_indices) < 7:
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

                # Sample or enumerate (k, j, l) triples
                # Constraint: i < k < j < l in circular order
                n_valid = len(valid_indices)

                if sample_size > 0:
                    # Sample combinations
                    n_samples = min(
                        sample_size,
                        (n_valid - 3) * (n_valid - 4) * (n_valid - 5) // 6,
                    )

                    # Random sampling with sorting to ensure k < j < l
                    samples = torch.randint(i_idx + 1, n_valid, (n_samples, 3), device=device)
                    samples, _ = torch.sort(samples, dim=1)
                    k_samples = samples[:, 0]
                    j_samples = samples[:, 1]
                    l_samples = samples[:, 2]

                    # Filter duplicates
                    valid_mask = (k_samples < j_samples) & (j_samples < l_samples)
                    k_samples = k_samples[valid_mask]
                    j_samples = j_samples[valid_mask]
                    l_samples = l_samples[valid_mask]
                else:
                    # Exhaustive search
                    k_list, j_list, l_list = [], [], []
                    for k_idx in range(i_idx + 1, n_valid - 2):
                        for j_idx in range(k_idx + 1, n_valid - 1):
                            for l_idx in range(j_idx + 1, n_valid):
                                k_list.append(k_idx)
                                j_list.append(j_idx)
                                l_list.append(l_idx)

                    if not k_list:
                        continue

                    k_samples = torch.tensor(k_list, device=device)
                    j_samples = torch.tensor(j_list, device=device)
                    l_samples = torch.tensor(l_list, device=device)

                if len(k_samples) == 0:
                    continue

                # Evaluate each (k, j, l) combination
                for k_idx_sample, j_idx_sample, l_idx_sample in zip(k_samples, j_samples, l_samples):
                    k = valid_indices[k_idx_sample].item()
                    j = valid_indices[j_idx_sample].item()
                    l = valid_indices[l_idx_sample].item()

                    v_k = tour[k].item()
                    v_j = tour[j].item()
                    v_l = tour[l].item()

                    k_next = (k + 1) % N
                    j_next = (j + 1) % N
                    l_next = (l + 1) % N

                    v_k_next = tour[k_next].item()
                    v_j_next = tour[j_next].item()
                    v_l_next = tour[l_next].item()

                    # Compute delta cost
                    # Removed edges
                    removed = (
                        dist[v_i_prev, v_i]
                        + dist[v_i, v_i_next]
                        + dist[v_k, v_k_next]
                        + dist[v_j, v_j_next]
                        + dist[v_l, v_l_next]
                    )

                    # Inserted edges (Type III pattern)
                    inserted = (
                        dist[v_i_prev, v_k] + dist[v_i_next, v_j] + dist[v_k_next, v_l] + dist[v_j_next, v_l_next]
                    )

                    delta = inserted - removed

                    if delta < best_delta - IMPROVEMENT_EPSILON:
                        best_delta = delta
                        best_move = (i, k, j, l)

            # Apply best move if found
            if best_move is not None:
                i, k, j, l = best_move

                # Apply Type III unstringing
                tour = _apply_type_iii_move(tour, i, k, j, l)
                tours[b] = tour
                improved = True

        if not improved:
            break

    return tours if is_batch else tours.squeeze(0)


def _apply_type_iii_move(
    tour: torch.Tensor,
    i: int,
    k: int,
    j: int,
    l: int,
) -> torch.Tensor:
    """
    Apply Type III unstringing move to a tour.

    Removes V_i and reconnects with Type III pattern:
    V_{i-1} -> S1_rev -> S2_rev -> S3_rev -> Remainder

    Where:
        S1 = V_{i+1} ... V_k
        S2 = V_{k+1} ... V_j
        S3 = V_{j+1} ... V_l
        Remainder = V_{l+1} ...

    Args:
        tour: Tour tensor [N]
        i: Index of node to remove
        k: Index of V_k
        j: Index of V_j
        l: Index of V_l

    Returns:
        torch.Tensor: Modified tour
    """
    n = len(tour)
    tour_list = tour.tolist()

    # Extract segments
    i_next = (i + 1) % n
    i_prev = (i - 1) if i > 0 else (n - 1)

    # S1: V_{i+1} ... V_k
    if i_next <= k:
        s1 = tour_list[i_next : k + 1]
    else:
        s1 = tour_list[i_next:] + tour_list[: k + 1]

    # S2: V_{k+1} ... V_j
    k_next = (k + 1) % n
    if k_next <= j:
        s2 = tour_list[k_next : j + 1]
    else:
        s2 = tour_list[k_next:] + tour_list[: j + 1]

    # S3: V_{j+1} ... V_l
    j_next = (j + 1) % n
    if j_next <= l:
        s3 = tour_list[j_next : l + 1]
    else:
        s3 = tour_list[j_next:] + tour_list[: l + 1]

    # Remainder: V_{l+1} ... V_{i-1}
    l_next = (l + 1) % n
    if l_next <= i_prev:
        remainder = tour_list[l_next : i_prev + 1]
    else:
        remainder = tour_list[l_next:] + tour_list[: i_prev + 1]

    # Reconstruct: V_{i-1} -> S1_rev -> S2_rev -> S3_rev -> Remainder
    v_i_prev = tour_list[i_prev]
    new_tour = [v_i_prev] + s1[::-1] + s2[::-1] + s3[::-1] + remainder

    # Restore depot at front if present
    if 0 in new_tour:
        depot_idx = new_tour.index(0)
        new_tour = new_tour[depot_idx:] + new_tour[:depot_idx]

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
