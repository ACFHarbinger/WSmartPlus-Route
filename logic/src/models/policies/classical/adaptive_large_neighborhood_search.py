"""
Vectorized Adaptive Large Neighborhood Search (ALNS) Policy.

This module implements a vectorized version of the ALNS algorithm for
Vehicle Routing Problems (VRP), optimized for GPU execution using PyTorch.
"""

import time

import torch

from .local_search import (
    vectorized_relocate,
    vectorized_swap,
    vectorized_swap_star,
    vectorized_three_opt,
    vectorized_two_opt,
    vectorized_two_opt_star,
)
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


def vectorized_cluster_removal(tours, dist_matrix, n_remove):
    """
    Removes a cluster of spatially related nodes.
    """
    B, N = tours.size()
    device = tours.device

    # 1. Pick a random seed node for each batch
    seed_idx = torch.randint(0, N, (B,), device=device)
    seed_nodes = torch.gather(tours, 1, seed_idx.unsqueeze(1)).squeeze(1)

    # 2. Find distances from seed nodes to all other nodes
    batch_indices = torch.arange(B, device=device).unsqueeze(1).expand(-1, N)
    tour_distances = dist_matrix[batch_indices, seed_nodes.unsqueeze(1), tours]

    # 3. Take n_remove nearest nodes
    _, remove_idx = torch.topk(tour_distances, n_remove, dim=1, largest=False)

    # 4. Create mask and partial tours
    mask = torch.ones_like(tours, dtype=torch.bool)
    mask[batch_indices[:, :n_remove], remove_idx] = False

    removed_nodes = torch.gather(tours, 1, remove_idx)
    partial_tours = tours.clone()
    partial_tours[~mask] = -1

    return partial_tours, removed_nodes


# -----------------------------
# Vectorized Repair Operators
# -----------------------------


def vectorized_greedy_insertion(partial_tours, removed_nodes, dist_matrix):
    """
    Inserts removed nodes into partial tours greedily.
    Optimized for giant tour representation where neighbors are non-null nodes.
    Args:
        partial_tours: (B, N) with some -1s or shorter?
        removed_nodes: (B, n_rem)
        dist_matrix: (B, N_all, N_all)
    """
    B, N = partial_tours.size()
    n_rem = removed_nodes.size(1)
    device = partial_tours.device
    batch_idx = torch.arange(B, device=device)

    current_tours = partial_tours.clone()

    for k in range(n_rem):
        nodes_to_insert = removed_nodes[:, k]

        # 1. Identify neighbors for every slot (-1)
        # Extend tours with virtual depots at start and end
        extended_tours = torch.cat(
            [
                torch.zeros((B, 1), dtype=torch.long, device=device),
                current_tours,
                torch.zeros((B, 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )

        # Find predecessors and successors for each position in the tour
        is_valid = extended_tours != -1
        val_indices = torch.arange(N + 2, device=device).unsqueeze(0).expand(B, -1)

        # prev_valid_idx: largest valid index <= current_idx
        prev_valid_idx = torch.cummax(val_indices * is_valid.long(), dim=1)[0]

        # next_valid_idx: smallest valid index >= current_idx
        # We use a large value for invalid nodes to find min from right
        large_val = N + 2
        shifted_indices = torch.where(is_valid, val_indices, torch.full_like(val_indices, large_val))
        # Flip, cummin, flip
        next_valid_idx = torch.flip(torch.cummin(torch.flip(shifted_indices, dims=[1]), dim=1)[0], dims=[1])

        p_idx = prev_valid_idx[:, :-2]
        n_idx = next_valid_idx[:, 2:]

        prev_nodes = torch.gather(extended_tours, 1, p_idx)
        next_nodes = torch.gather(extended_tours, 1, n_idx)

        # 2. Compute insertion cost for all possible slots (-1 positions)
        cost_matrix = (
            dist_matrix[batch_idx.unsqueeze(1), prev_nodes, nodes_to_insert.unsqueeze(1)]
            + dist_matrix[batch_idx.unsqueeze(1), nodes_to_insert.unsqueeze(1), next_nodes]
            - dist_matrix[batch_idx.unsqueeze(1), prev_nodes, next_nodes]
        )

        # Mask out slots that AREN'T -1
        is_hole = current_tours == -1
        cost_matrix[~is_hole] = float("inf")

        # 3. Find best slot for each instance
        _, best_slots = torch.min(cost_matrix, dim=1)

        # 4. Insert node
        current_tours[batch_idx, best_slots] = nodes_to_insert

    return current_tours


def vectorized_regret_2_insertion(partial_tours, removed_nodes, dist_matrix):
    """
    Inserts nodes based on Regret-2 criterion: Maximize (cost_2nd_best - cost_best).
    """
    B, N = partial_tours.size()
    n_rem = removed_nodes.size(1)
    device = partial_tours.device
    batch_idx = torch.arange(B, device=device)

    current_tours = partial_tours.clone()
    pending_nodes = removed_nodes.clone()
    is_pending = torch.ones((B, n_rem), dtype=torch.bool, device=device)

    for _ in range(n_rem):
        # 1. Compute neighbors for all slots
        extended_tours = torch.cat(
            [
                torch.zeros((B, 1), dtype=torch.long, device=device),
                current_tours,
                torch.zeros((B, 1), dtype=torch.long, device=device),
            ],
            dim=1,
        )

        is_valid = extended_tours != -1
        val_indices = torch.arange(N + 2, device=device).unsqueeze(0).expand(B, -1)

        # Largest valid index <= current_idx
        prev_valid_idx = torch.cummax(val_indices * is_valid.long(), dim=1)[0]

        # Smallest valid index >= current_idx
        large_val = N + 2
        shifted_indices = torch.where(is_valid, val_indices, torch.full_like(val_indices, large_val))
        next_valid_idx = torch.flip(torch.cummin(torch.flip(shifted_indices, dims=[1]), dim=1)[0], dims=[1])

        p_idx = prev_valid_idx[:, :-2]
        n_idx = next_valid_idx[:, 2:]
        prev_nodes = torch.gather(extended_tours, 1, p_idx)
        next_nodes = torch.gather(extended_tours, 1, n_idx)

        # 2. For each pending node, calculate best and 2nd best insertion costs
        # This is (B, n_rem, N) costs
        # Only for nodes where is_pending is True

        # (B, 1, N)
        prev_exp = prev_nodes.unsqueeze(1)
        next_exp = next_nodes.unsqueeze(1)

        # (B, n_rem, 1)
        nodes_exp = pending_nodes.unsqueeze(2)

        # (B, n_rem, N)
        all_costs = (
            dist_matrix[batch_idx.view(B, 1, 1), prev_exp, nodes_exp]
            + dist_matrix[batch_idx.view(B, 1, 1), nodes_exp, next_exp]
            - dist_matrix[batch_idx.view(B, 1, 1), prev_exp, next_exp]
        )

        # Mask out non-holes (B, 1, N)
        is_hole = (current_tours == -1).unsqueeze(1)
        all_costs[~is_hole.expand(-1, n_rem, -1)] = 1e9

        # (B, n_rem) best costs and their slot indices
        best_costs, best_slots = torch.min(all_costs, dim=2)

        # To find second best, mask the best and min again
        all_costs.scatter_(2, best_slots.unsqueeze(2), 1e9)
        second_best_costs, _ = torch.min(all_costs, dim=2)

        # Regret = 2nd_best - best
        regret = second_best_costs - best_costs

        # Mask out already inserted nodes
        regret[~is_pending] = -1e9

        # 3. Pick node with max regret
        _, best_node_idx = torch.max(regret, dim=1)

        # 4. Insert best nodes into their best slots
        insert_nodes = torch.gather(pending_nodes, 1, best_node_idx.unsqueeze(1)).squeeze(1)
        insert_slots = torch.gather(best_slots, 1, best_node_idx.unsqueeze(1)).squeeze(1)

        current_tours[batch_idx, insert_slots] = insert_nodes
        is_pending.scatter_(1, best_node_idx.unsqueeze(1), False)

    return current_tours


class VectorizedALNS:
    """
    Vectorized Adaptive Large Neighborhood Search Solver.
    """

    def __init__(self, dist_matrix, demands, vehicle_capacity, time_limit=1.0, device="cuda"):
        """
        Initialize the Vectorized ALNS solver.

        Args:
            dist_matrix: Distance matrix [B, N, N].
            demands: Node demands [B, N].
            vehicle_capacity: Vehicle capacity constraint.
            time_limit: Time limit for solving in seconds.
            device: Computation device ('cpu' or 'cuda').
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.time_limit = time_limit
        self.device = device

        self.destroy_ops = [vectorized_random_removal, vectorized_worst_removal, vectorized_cluster_removal]
        self.repair_ops = [vectorized_greedy_insertion, vectorized_regret_2_insertion]

        self.d_weights = torch.ones(len(self.destroy_ops), device=device)
        self.r_weights = torch.ones(len(self.repair_ops), device=device)

    def solve(
        self, initial_solutions, n_iterations=2000, time_limit=None, max_vehicles=0, start_temp=0.5, cooling_rate=0.9995
    ):
        """
        High-performance ALNS Solve.
        Optimizes the giant tour as a TSP and splits only at the end.
        """
        B, N = initial_solutions.size()
        device = initial_solutions.device
        start_time = time.time()

        current_solutions = initial_solutions.clone()

        # Initial TSP costs (dist_matrix[prev, curr])
        def get_tsp_costs(tours):
            padded = torch.zeros(B, N + 2, dtype=torch.long, device=device)
            padded[:, 1:-1] = tours
            shifted = torch.roll(padded, shifts=-1, dims=1)
            dist = self.dist_matrix[torch.arange(B, device=device).unsqueeze(1), padded, shifted]
            return dist[:, :-1].sum(dim=1)

        current_costs = get_tsp_costs(current_solutions)
        best_solutions = current_solutions.clone()
        best_costs = current_costs.clone()

        T = torch.full((B,), float(start_temp), device=device)

        for i in range(n_iterations):
            if time_limit and (time.time() - start_time > time_limit):
                break

            # 1. Select Operators
            d_idx = int(torch.multinomial(self.d_weights, 1).item())
            r_idx = int(torch.multinomial(self.r_weights, 1).item())
            destroy_op = self.destroy_ops[d_idx]
            repair_op = self.repair_ops[r_idx]

            n_remove = torch.randint(max(1, int(N * 0.1)), max(2, int(N * 0.4) + 1), (1,)).item()

            # 2. Destroy & Repair (Fast TSP operators)
            if destroy_op == vectorized_random_removal:
                partial, removed = destroy_op(current_solutions, n_remove)
            else:
                partial, removed = destroy_op(current_solutions, self.dist_matrix, n_remove)

            candidate_solutions = repair_op(partial, removed, self.dist_matrix)

            # 3. Fast Education (TSP Local Search)
            if i % 20 == 0:
                candidate_solutions = vectorized_two_opt(candidate_solutions, self.dist_matrix, max_iterations=20)
                candidate_solutions = vectorized_relocate(candidate_solutions, self.dist_matrix, max_iterations=10)
                candidate_solutions = vectorized_swap(candidate_solutions, self.dist_matrix, max_iterations=10)
                if N > 40 and i % 100 == 0:
                    candidate_solutions = vectorized_three_opt(candidate_solutions, self.dist_matrix, max_iterations=5)

            # 3b. Advanced Inter-Route Education (Periodic)
            if i > 0 and i % 100 == 0:
                # 1. Split into CVRP routes (returns list of lists)
                routes_list, _ = vectorized_linear_split(
                    candidate_solutions,
                    self.dist_matrix,
                    self.demands,
                    self.vehicle_capacity,
                    max_vehicles=max_vehicles,
                )

                # 2. Convert to padded tensor for vectorized operators
                max_r_len = max(len(r) for r in routes_list)
                routes_tensor = torch.zeros((B, max_r_len), device=device, dtype=torch.long)
                for b_idx in range(B):
                    r_nodes = torch.tensor(routes_list[b_idx], device=device, dtype=torch.long)
                    routes_tensor[b_idx, : len(r_nodes)] = r_nodes

                # 3. Apply inter-route operators
                routes_tensor = vectorized_two_opt_star(routes_tensor, self.dist_matrix, max_iterations=10)
                routes_tensor = vectorized_swap_star(routes_tensor, self.dist_matrix, max_iterations=10)

                # 4. Flatten back to giant tour
                flattened = []
                for b_idx in range(B):
                    # extract non-zero nodes (customers)
                    nodes = routes_tensor[b_idx]
                    mask_nz = nodes != 0
                    flat_nodes = nodes[mask_nz]
                    # Ensure it has exactly N customers (it should if split was complete)
                    if flat_nodes.size(0) < N:
                        # Re-attach missing nodes if any or pad
                        existing = set(flat_nodes.tolist())
                        all_c = set(candidate_solutions[b_idx].tolist())
                        missing = list(all_c - existing)
                        if missing:
                            flat_nodes = torch.cat([flat_nodes, torch.tensor(missing, device=device, dtype=torch.long)])
                        if flat_nodes.size(0) < N:
                            flat_nodes = torch.cat(
                                [flat_nodes, torch.zeros(N - flat_nodes.size(0), device=device, dtype=torch.long)]
                            )
                    elif flat_nodes.size(0) > N:
                        flat_nodes = flat_nodes[:N]
                    flattened.append(flat_nodes)
                candidate_solutions = torch.stack(flattened)

            # 4. Evaluate TSP gain
            candidate_costs = get_tsp_costs(candidate_solutions)
            delta = candidate_costs - current_costs

            # 5. Accept/Reject (SA)
            prob = torch.exp(-delta / T).clamp(max=1.0)
            accept = (delta <= 0) | (torch.rand(B, device=device) < prob)

            current_solutions[accept] = candidate_solutions[accept]
            current_costs[accept] = candidate_costs[accept]

            improved = candidate_costs < best_costs
            best_solutions[improved] = candidate_solutions[improved]
            best_costs[improved] = candidate_costs[improved]

            # 6. Weights
            scores = torch.zeros(B, device=device)
            scores[improved] = 10.0
            scores[accept & ~improved] = 2.0
            avg_score = scores.mean().item()
            self.d_weights[d_idx] = 0.9 * self.d_weights[d_idx] + 0.1 * avg_score
            self.r_weights[r_idx] = 0.9 * self.r_weights[r_idx] + 0.1 * avg_score

            T *= cooling_rate

        # Final conversion to CVRP routes using optimal split
        best_routes, final_costs = vectorized_linear_split(
            best_solutions, self.dist_matrix, self.demands, self.vehicle_capacity, max_vehicles=max_vehicles
        )

        return best_routes, final_costs
