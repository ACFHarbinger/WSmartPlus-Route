"""
Vectorized Adaptive Large Neighborhood Search (ALNS) Policy.

This module implements a vectorized version of the ALNS algorithm for
Vehicle Routing Problems (VRP), optimized for GPU execution using PyTorch.
"""

import time

import torch

from .split import vectorized_linear_split

# -----------------------------
# Vectorized Destroy Operators
# -----------------------------


def vectorized_random_removal(tours, n_remove):
    """
    Randomly removes nodes from tours.
    Args:
        tours: (B, N) tensor of giant tours (1..N nodes)
        n_remove: number of nodes to remove
    Returns:
        partial_tours: (B, N) with removed nodes set to -1
        removed_nodes: (B, n_remove)
    """
    B, N = tours.size()
    device = tours.device

    # Generate random indices for each batch
    # We want to remove nodes (not indices 0..N-1)
    # But it's easier to remove indices in the giant tour

    # (B, N) random values
    rand = torch.rand(B, N, device=device)
    # Sort and take top n_remove
    _, remove_idx = torch.topk(rand, n_remove, dim=1)

    # Create mask
    mask = torch.ones_like(tours, dtype=torch.bool)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_remove)
    mask[batch_indices, remove_idx] = False

    removed_nodes = torch.gather(tours, 1, remove_idx)
    partial_tours = tours.clone()
    partial_tours[~mask] = -1

    return partial_tours, removed_nodes


def vectorized_worst_removal(tours, dist_matrix, n_remove, p=3):
    """
    Removes nodes that contribute most to the total distance.
    Args:
        tours: (B, N) tensor
        dist_matrix: (B, N_all, N_all)
        n_remove: int
        p: determinism parameter
    """
    B, N = tours.size()
    device = tours.device

    # Compute removal cost for each node in current tour
    # Node at pos i: cost = dist(pos-1, i) + dist(i, pos+1) - dist(pos-1, pos+1)

    # Pad tours to handle boundaries (depot is 0)
    padded = torch.zeros(B, N + 2, dtype=torch.long, device=device)
    padded[:, 1:-1] = tours

    prev_nodes = padded[:, :-2]
    curr_nodes = padded[:, 1:-1]
    next_nodes = padded[:, 2:]

    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)

    costs = (
        dist_matrix[batch_idx, prev_nodes, curr_nodes]
        + dist_matrix[batch_idx, curr_nodes, next_nodes]
        - dist_matrix[batch_idx, prev_nodes, next_nodes]
    )

    # Randomized worst selection
    # We sample indices based on rank ^ p or similar
    # For simplicity, we just take actual top costs with some noise or use p
    sorted_costs, sorted_idx = torch.sort(costs, dim=1, descending=True)

    # If p is large, we take the top ones.
    # To simulate the "randomized" part, we can shuffle slightly or use topk on (costs * rand^1/p)
    noise = torch.rand_like(costs)
    noised_costs = costs * (noise ** (1 / p))
    _, remove_idx = torch.topk(noised_costs, n_remove, dim=1)

    mask = torch.ones_like(tours, dtype=torch.bool)
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, n_remove)
    mask[batch_indices, remove_idx] = False

    removed_nodes = torch.gather(tours, 1, remove_idx)
    partial_tours = tours.clone()
    partial_tours[~mask] = -1

    return partial_tours, removed_nodes


# -----------------------------
# Vectorized Repair Operators
# -----------------------------


def vectorized_greedy_insertion(partial_tours, removed_nodes, dist_matrix):
    """
    Inserts removed nodes into partial tours greedily based on permutation cost.
    Args:
        partial_tours: (B, N) with some -1s or shorter?
        removed_nodes: (B, n_rem)
        dist_matrix: (B, N_all, N_all)
    """
    B, N = partial_tours.size()
    B_rem, n_rem = removed_nodes.size()
    device = partial_tours.device

    # current_tours will hold nodes and -1s
    current_tours = partial_tours.clone()

    for k in range(n_rem):
        nodes_to_insert = removed_nodes[:, k]  # (B,)

        # We need to find the best slot among remaining -1s.
        # For simplicity in vectorization, we identify all current "available" slots.
        # Available slots are positions currently holding -1.

        # To make it truly greedy in the permutation sense:
        # We look at all pairs of nodes (i, j) that are currently "neighbors"
        # in the partial tour (skipping -1s).

        # This is complex. Let's use a simpler heuristic:
        # Fill the first -1 but skip if needed? No, let's just find the best slot.
        # A slot is between two non -1 nodes.

        # Simpler: Just fill -1s in order for now, but ALNS usually
        # picks the node that has the best insertion cost.

        # Re-implementing: Find first -1 and put it there.
        # To be "Greedy", we should try all -1 positions for a node.

        mask = current_tours == -1
        batch_idx = torch.arange(B, device=device)

        # Find all candidate positions (indices of -1)
        # We'll just pick the one that minimizes dist(prev, u) + dist(u, next)

        # For each batch, find indices of -1
        # This is non-uniform across batch if we were clever, but here it's uniform n_rem - k
        # We can use topk or similar to get all indices.
        _, slots = torch.topk(mask.float(), n_rem - k, dim=1)  # (B, remaining_slots)

        # Try all slots for the current node
        best_slots = torch.zeros(B, dtype=torch.long, device=device)
        min_costs = torch.full((B,), float("inf"), device=device)

        for s_idx in range(slots.size(1)):
            slot = slots[:, s_idx]  # (B,)

            # Neighbors of this slot in current_tours
            # Note: current_tours might have other -1s.
            # We look for the nearest non-minus-one nodes.

            # Pre-slot node
            l_idx = slot - 1
            while l_idx.min() >= 0:
                # This loop is problematic for vectorization.
                break

            # Let's simplify: Use the immediate neighbors in the fixed-size array.
            # If neighbor is -1, treat as depot (0) or ignore.
            prev_idx = torch.clamp(slot - 1, min=0)
            next_idx = torch.clamp(slot + 1, max=N - 1)

            prev_nodes = torch.gather(current_tours, 1, prev_idx.unsqueeze(1)).squeeze(1)
            next_nodes = torch.gather(current_tours, 1, next_idx.unsqueeze(1)).squeeze(1)

            # Replace -1 with 0 (depot)
            prev_nodes[prev_nodes == -1] = 0
            next_nodes[next_nodes == -1] = 0

            cost = (
                dist_matrix[batch_idx, prev_nodes, nodes_to_insert]
                + dist_matrix[batch_idx, nodes_to_insert, next_nodes]
                - dist_matrix[batch_idx, prev_nodes, next_nodes]
            )

            better = cost < min_costs
            min_costs[better] = cost[better]
            best_slots[better] = slot[better]

        current_tours[batch_idx, best_slots] = nodes_to_insert

    return current_tours


class VectorizedALNS:
    """
    Vectorized Adaptive Large Neighborhood Search Solver.
    """

    def __init__(self, dist_matrix, demands, vehicle_capacity, time_limit=1.0, device="cuda"):
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.time_limit = time_limit
        self.device = device

        self.destroy_ops = [vectorized_random_removal, vectorized_worst_removal]
        self.repair_ops = [vectorized_greedy_insertion]

        self.d_weights = torch.ones(len(self.destroy_ops), device=device)
        self.r_weights = torch.ones(len(self.repair_ops), device=device)

    def solve(self, initial_solutions, n_iterations=100, time_limit=None):
        B, N = initial_solutions.size()
        start_time = time.time()

        current_solutions = initial_solutions.clone()
        _, current_costs = vectorized_linear_split(
            current_solutions, self.dist_matrix, self.demands, self.vehicle_capacity
        )
        best_solutions = current_solutions.clone()
        best_costs = current_costs.clone()

        # Simulated annealing params
        T = 1.0
        cooling_rate = (0.01 / T) ** (1 / n_iterations) if n_iterations > 0 else 0.99

        for i in range(n_iterations):
            if time_limit and (time.time() - start_time > time_limit):
                break

            # 1. Select operators based on weights
            d_idx = torch.multinomial(self.d_weights, 1).item()
            r_idx = torch.multinomial(self.r_weights, 1).item()

            d_op = self.destroy_ops[d_idx]
            r_op = self.repair_ops[r_idx]

            n_remove = max(1, int(N * 0.2))

            # 2. Destroy
            if d_op == vectorized_worst_removal:
                partial, removed = d_op(current_solutions, self.dist_matrix, n_remove)
            else:
                partial, removed = d_op(current_solutions, n_remove)

            # 3. Repair
            candidate_solutions = r_op(partial, removed, self.dist_matrix)

            # 4. Evaluate
            _, candidate_costs = vectorized_linear_split(
                candidate_solutions, self.dist_matrix, self.demands, self.vehicle_capacity
            )

            # 5. Accept/Reject
            delta = candidate_costs - current_costs

            # Accept if better or by SA probability
            accept_prob = torch.exp(-delta / T).clamp(max=1.0)
            rand_vals = torch.rand(B, device=self.device)
            accept_mask = (delta < 0) | (rand_vals < accept_prob)

            current_solutions[accept_mask] = candidate_solutions[accept_mask]
            current_costs[accept_mask] = candidate_costs[accept_mask]

            # Update best
            improved_mask = candidate_costs < best_costs
            best_solutions[improved_mask] = candidate_solutions[improved_mask]
            best_costs[improved_mask] = candidate_costs[improved_mask]

            # Update weights based on score
            # score: 10 if new best, 5 if improved current, 2 if accepted
            scores = torch.zeros(B, device=self.device)
            scores[accept_mask] = 2.0
            scores[delta < 0] = 5.0
            scores[improved_mask] = 10.0

            # Average score for this operator pair
            avg_score = scores.mean()
            # Adaptive update
            decay = 0.9
            self.d_weights[d_idx] = decay * self.d_weights[d_idx] + (1 - decay) * avg_score
            self.r_weights[r_idx] = decay * self.r_weights[r_idx] + (1 - decay) * avg_score

            T *= cooling_rate

        # Final split to get routes
        best_routes, best_costs = vectorized_linear_split(
            best_solutions, self.dist_matrix, self.demands, self.vehicle_capacity
        )

        return best_routes, best_costs
