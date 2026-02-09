"""
Vectorized Lin-Kernighan-Helsgaun (LKH-3) Operator.

This module provides a GPU-accelerated implementation of the sophisticated LKH-3
local search algorithm. LKH combines powerful k-opt moves with alpha-nearness
candidate sets and lexicographic optimization for VRP problems.

Key features:
- Alpha-measure edge pruning based on MST
- Candidate set restriction for efficiency
- Sequential 2-opt and 3-opt moves
- Double bridge kicks for perturbation
- Lexicographic optimization (penalty, cost)
"""

from typing import List, Optional, Tuple

import torch
from logic.src.constants.routing import IMPROVEMENT_EPSILON


def vectorized_lkh(
    tours: torch.Tensor,
    distance_matrix: torch.Tensor,
    capacities: Optional[torch.Tensor] = None,
    demands: Optional[torch.Tensor] = None,
    max_iterations: int = 100,
    max_candidates: int = 5,
    use_3opt: bool = True,
    perturbation_interval: int = 10,
) -> torch.Tensor:
    """
    Vectorized Lin-Kernighan-Helsgaun local search across a batch of tours using PyTorch.

    LKH is one of the most sophisticated local search algorithms for TSP/VRP, combining:
    1. Alpha-nearness for edge pruning (based on MST)
    2. Sequential k-opt moves (2-opt and 3-opt)
    3. Double bridge perturbations for escaping local optima
    4. Lexicographic optimization: minimize (penalty, cost)

    The algorithm iteratively improves tours by:
    - Selecting promising edge exchanges based on candidate sets
    - Applying 2-opt moves for quick improvements
    - Extending to 3-opt for deeper search
    - Periodically applying double bridge kicks

    Algorithm (per batch instance):
    1. Compute alpha measures based on MST
    2. Build candidate sets (5-10 nearest neighbors per node)
    3. Local search phase:
        a. For each edge (t1, t2):
            - Check candidate neighbors t3 for t2
            - Evaluate 2-opt gain: d(t1,t2) + d(t3,t4) - d(t1,t3) - d(t2,t4)
            - If positive gain and feasible: apply move
            - Optionally extend to 3-opt
        b. Repeat until local optimum
    4. Perturbation: Apply double bridge (4-opt) kick
    5. Repeat steps 3-4 for max_iterations

    Args:
        tours: Batch of tours [B, N] where B=batch size, N=tour length
            Note: Tours should be closed (start and end with depot 0)
        distance_matrix: Pairwise distances [B, N+1, N+1] or [N+1, N+1] (shared)
        capacities: Vehicle capacities [B] or scalar (optional, for penalty computation)
        demands: Node demands [B, N+1] or [N+1] (optional, for penalty computation)
        max_iterations: Number of ILS iterations (local search + kick cycles)
        max_candidates: Number of candidate edges per node (default: 5)
            Higher = more thorough but slower. LKH-3 typically uses 5-10.
        use_3opt: Whether to use 3-opt extensions (default: True)
            Disable for faster but less thorough search
        perturbation_interval: Apply kicks every N iterations (default: 10)

    Returns:
        torch.Tensor: Improved tours [B, N] with same shape as input

    Note:
        - Tours must have at least 8 nodes for meaningful LKH moves
        - Complexity: O(N² × max_candidates) per local search phase
        - 3-opt adds O(N³) but is limited by candidate sets
        - This is a hybrid implementation: batch parallelism + instance-level iteration
        - For best results, use max_iterations >= 50

    Mathematical Formulation:
        2-opt gain: G₂ = d(t₁,t₂) + d(t₃,t₄) - d(t₁,t₃) - d(t₂,t₄)
        3-opt gain: G₃ = d(t₁,t₂) + d(t₃,t₄) + d(t₅,t₆)
                        - d(t₁,t₃) - d(t₂,t₅) - d(t₄,t₆)

        Lexicographic ordering: (p₁, c₁) < (p₂, c₂) ⟺ p₁ < p₂ ∨ (p₁ = p₂ ∧ c₁ < c₂)

        Alpha measure: α(i,j) = c(i,j) - max_{e∈MST_path(i,j)} c(e)

    Example:
        >>> tours = torch.tensor([[0, 5, 3, 8, 2, 7, 1, 4, 6, 0]])
        >>> dist = torch.rand(10, 10)
        >>> improved = vectorized_lkh(tours, dist, max_iterations=50)
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

    if N < 8:  # LKH needs sufficient nodes for meaningful moves
        return tours if is_batch else tours.squeeze(0)

    # Expand distance matrix if shared
    if distance_matrix.size(0) == 1 and B > 1:
        distance_matrix = distance_matrix.expand(B, -1, -1)

    # Handle demands and capacities
    has_capacity = demands is not None and capacities is not None

    if has_capacity:
        if demands.dim() == 1:
            demands = demands.unsqueeze(0).expand(B, -1)
        if capacities.dim() == 0:
            capacities = capacities.unsqueeze(0).expand(B)
        elif capacities.dim() == 1 and capacities.size(0) != B:
            capacities = capacities.expand(B)

    # Main ILS loop
    for iteration in range(max_iterations):
        # Process each batch instance
        for b in range(B):
            tour = tours[b]
            dist = distance_matrix[b]

            # Compute alpha measures and candidate sets
            alpha = _compute_alpha_measures(dist, device)
            candidates = _get_candidate_sets(alpha, max_candidates)

            # Capacity constraints
            if has_capacity:
                demand = demands[b]
                capacity = capacities[b]
            else:
                demand = None
                capacity = None

            # Local search phase: iterate until local optimum
            local_improved = True
            while local_improved:
                local_improved = False

                # Get current score
                curr_penalty, curr_cost = _compute_score(tour, dist, demand, capacity)

                # Try 2-opt moves for each edge
                nodes_count = len(tour) - 1  # Exclude duplicate depot at end

                for i in range(nodes_count):
                    t1 = tour[i].item()
                    t2 = tour[i + 1].item()

                    # Check candidates for t2
                    for t3 in candidates[t2]:
                        if t3 == t1:
                            continue

                        # Find t3 in tour
                        t3_positions = (tour == t3).nonzero(as_tuple=True)[0]
                        if len(t3_positions) == 0:
                            continue

                        j = t3_positions[0].item()

                        # Skip if adjacent or invalid
                        if j <= i + 1 or j >= nodes_count:
                            continue

                        t4 = tour[j + 1].item()

                        # Compute 2-opt gain
                        gain = (dist[t1, t2] + dist[t3, t4]) - (dist[t1, t3] + dist[t2, t4])

                        if gain > IMPROVEMENT_EPSILON:
                            # Apply 2-opt move
                            new_tour = _apply_2opt(tour, i, j)
                            new_penalty, new_cost = _compute_score(new_tour, dist, demand, capacity)

                            # Lexicographic comparison
                            if _is_better(new_penalty, new_cost, curr_penalty, curr_cost):
                                tour = new_tour
                                curr_penalty = new_penalty
                                curr_cost = new_cost
                                local_improved = True
                                break

                        # Try 3-opt extension
                        if use_3opt and not local_improved:
                            for k in range(j + 2, min(j + 20, nodes_count)):  # Limit search
                                t5 = tour[k].item()
                                t6 = tour[k + 1].item()

                                # Compute 3-opt gain
                                gain3 = (dist[t1, t2] + dist[t3, t4] + dist[t5, t6]) - (
                                    dist[t1, t3] + dist[t2, t5] + dist[t4, t6]
                                )

                                if gain3 > IMPROVEMENT_EPSILON:
                                    new_tour = _apply_3opt(tour, i, j, k)
                                    new_penalty, new_cost = _compute_score(new_tour, dist, demand, capacity)

                                    if _is_better(new_penalty, new_cost, curr_penalty, curr_cost):
                                        tour = new_tour
                                        curr_penalty = new_penalty
                                        curr_cost = new_cost
                                        local_improved = True
                                        break

                    if local_improved:
                        break

            # Update tour in batch
            tours[b] = tour

        # Apply perturbation periodically
        if (iteration + 1) % perturbation_interval == 0 and iteration < max_iterations - 1:
            for b in range(B):
                tours[b] = _double_bridge_kick(tours[b])

    return tours if is_batch else tours.squeeze(0)


def _compute_alpha_measures(
    distance_matrix: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute alpha measures for edge pruning based on MST.

    Alpha(i,j) ≈ distance(i,j) for non-MST edges, 0 for MST edges.
    This is a simplified version; full LKH uses path-based alpha computation.

    Args:
        distance_matrix: Distance matrix [N, N]
        device: Target device

    Returns:
        Alpha measures [N, N]
    """
    # Simplified: Use Prim's algorithm approximation
    # For full implementation, would compute MST and path-based alpha
    n = distance_matrix.size(0)
    alpha = distance_matrix.clone()

    # Find MST edges (simplified)
    # In full LKH-3, this uses path-based computation
    # Here we approximate: nearby edges get lower alpha
    k = min(5, n - 1)
    _, indices = torch.topk(distance_matrix, k, dim=1, largest=False)

    # Set alpha to 0 for k-nearest neighbors (MST approximation)
    for i in range(n):
        for j in indices[i]:
            if j != i:
                alpha[i, j] = 0.0

    return alpha


def _get_candidate_sets(
    alpha_measures: torch.Tensor,
    max_candidates: int,
) -> List[List[int]]:
    """
    Generate candidate sets based on alpha measures.

    Args:
        alpha_measures: Alpha measures [N, N]
        max_candidates: Number of candidates per node

    Returns:
        List of candidate lists per node
    """
    n = alpha_measures.size(0)
    candidates = []

    for i in range(n):
        # Sort by alpha (ascending)
        sorted_indices = torch.argsort(alpha_measures[i])

        # Filter self and limit
        valid = [idx.item() for idx in sorted_indices if idx != i]
        candidates.append(valid[:max_candidates])

    return candidates


def _compute_score(
    tour: torch.Tensor,
    distance_matrix: torch.Tensor,
    demands: Optional[torch.Tensor],
    capacity: Optional[torch.Tensor],
) -> Tuple[float, float]:
    """
    Compute (penalty, cost) for lexicographic comparison.

    Args:
        tour: Tour [N]
        distance_matrix: Distance matrix [N, N]
        demands: Node demands [N] (optional)
        capacity: Vehicle capacity (scalar, optional)

    Returns:
        (penalty, cost) tuple
    """
    # Compute cost
    n = len(tour) - 1
    cost = 0.0
    for i in range(n):
        cost += distance_matrix[tour[i], tour[i + 1]].item()

    # Compute penalty (capacity violations)
    penalty = 0.0
    if demands is not None and capacity is not None:
        current_load = 0.0
        for node in tour:
            node_idx = node.item()
            if node_idx == 0:
                current_load = 0.0
            else:
                if node_idx < len(demands):
                    current_load += demands[node_idx].item()
                    if current_load > capacity.item() + 1e-6:
                        penalty += current_load - capacity.item()

    return penalty, cost


def _is_better(p1: float, c1: float, p2: float, c2: float) -> bool:
    """
    Lexicographic comparison: (p1, c1) < (p2, c2)?

    Args:
        p1, c1: Penalty and cost for solution 1
        p2, c2: Penalty and cost for solution 2

    Returns:
        True if solution 1 is better
    """
    if abs(p1 - p2) > 1e-6:
        return p1 < p2
    return c1 < c2 - 1e-6


def _apply_2opt(tour: torch.Tensor, i: int, j: int) -> torch.Tensor:
    """
    Apply 2-opt move: reverse segment [i+1, j].

    Args:
        tour: Tour tensor [N]
        i: Start index
        j: End index

    Returns:
        New tour with reversed segment
    """
    new_tour = tour.clone()
    new_tour[i + 1 : j + 1] = torch.flip(new_tour[i + 1 : j + 1], dims=[0])
    return new_tour


def _apply_3opt(tour: torch.Tensor, i: int, j: int, k: int) -> torch.Tensor:
    """
    Apply 3-opt move: reverse segments [i+1, j] and [j+1, k].

    Args:
        tour: Tour tensor [N]
        i: First break point
        j: Second break point
        k: Third break point

    Returns:
        New tour with reversed segments
    """
    new_tour = tour.clone()
    # Reverse first segment
    new_tour[i + 1 : j + 1] = torch.flip(new_tour[i + 1 : j + 1], dims=[0])
    # Reverse second segment
    new_tour[j + 1 : k + 1] = torch.flip(new_tour[j + 1 : k + 1], dims=[0])
    return new_tour


def _double_bridge_kick(tour: torch.Tensor) -> torch.Tensor:
    """
    Apply double bridge perturbation (4-opt move).

    Breaks 4 edges and reconnects segments in a different order.
    This is a powerful perturbation for escaping local optima.

    Args:
        tour: Tour tensor [N]

    Returns:
        Perturbed tour
    """
    n = len(tour) - 1  # Active nodes (excluding duplicate depot)

    if n < 8:
        return tour

    # Select 4 random positions
    positions = torch.randperm(n - 2) + 1  # Avoid first and last
    positions = positions[:4].sort()[0]
    a, b, c, d = positions.tolist()

    # Reconnect segments: [0..a] -> [c+1..d] -> [b+1..c] -> [a+1..b] -> [d+1..end]
    tour_list = tour.tolist()
    new_tour = (
        tour_list[: a + 1]
        + tour_list[c + 1 : d + 1]
        + tour_list[b + 1 : c + 1]
        + tour_list[a + 1 : b + 1]
        + tour_list[d + 1 :]
    )

    return torch.tensor(new_tour, dtype=tour.dtype, device=tour.device)
