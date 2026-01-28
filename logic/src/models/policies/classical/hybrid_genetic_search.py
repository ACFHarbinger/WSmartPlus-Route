"""
Vectorized Hybrid Genetic Search (HGS) Policy.

This module implements a vectorized version of the Hybrid Genetic Search algorithm for
Vehicle Routing Problems (VRP), optimized for GPU execution using PyTorch.

It includes:
- `VectorizedPopulation`: Management of the genetic population.
- `VectorizedHGS`: The main solver class.
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
from .split import (
    vectorized_linear_split,
)


# -----------------------------
# Vectorized Genetic Operators
# -----------------------------
def vectorized_ordered_crossover(parent1, parent2):
    """
    Vectorized Ordered Crossover (OX1) with shared cuts across batch.
    Args:
        parent1: (B, N)
        parent2: (B, N)
    Returns:
        offspring: (B, N)
    """
    B, N = parent1.size()
    device = parent1.device

    # 1. Generate shared cut points
    idx1, idx2 = torch.randint(0, N, (2,)).tolist()
    start = min(idx1, idx2)
    end = max(idx1, idx2)

    if start == end:
        end = min(start + 1, N)

    num_seg = end - start
    num_rem = N - num_seg

    # 2. Extract segment from Parent 1
    # segment: (B, num_seg)
    segment = parent1[:, start:end]

    # 3. Create Offspring container
    offspring = torch.zeros_like(parent1)
    # Copy segment
    offspring[:, start:end] = segment

    # 4. Fill remaining from Parent 2
    # We need to take elements from P2 that are NOT in segment, maintaining order
    # starting from 'end' index (wrapping around).

    # Roll P2 so it starts at 'end'
    # logical shift: indices [end, end+1, ..., N-1, 0, ..., end-1]
    roll_idx = torch.arange(N, device=device)
    roll_idx = (roll_idx + end) % N
    p2_rolled = parent2[:, roll_idx]  # (B, N) sorted by fill order

    # Identification of valid elements
    # Since values are 1..N (or 0..N-1), we can use a mask.
    # Naive check: (B, N, 1) == (B, 1, num_seg) -> any(dim=2)
    # segment: (B, num_seg)
    # p2_rolled: (B, N)

    # Efficient exclusion check:
    # (B, N, 1) == (B, 1, num_seg) -> (B, N, num_seg) -> sum/any -> (B, N) mask
    # This uses B*N*num_seg memory. For N=100, B=128, this is ~1.2M elements (bool). Cheap.

    exists_in_seg = (p2_rolled.unsqueeze(2) == segment.unsqueeze(1)).any(dim=2)  # (B, N)

    # We want elements where ~exists_in_seg
    # Handle each batch element separately to be robust to duplicates
    fill_idx = torch.cat([torch.arange(end, N, device=device), torch.arange(0, start, device=device)])

    for b in range(B):
        valid_mask = ~exists_in_seg[b]
        valid_vals_b = p2_rolled[b][valid_mask]

        # Pad or truncate to match num_rem
        if len(valid_vals_b) < num_rem:
            # Pad with values from parent1 that aren't in segment
            missing = num_rem - len(valid_vals_b)
            # Use first missing values from parent1 outside segment
            extra = parent1[b][
                torch.cat(
                    [
                        torch.arange(0, start, device=device),
                        torch.arange(end, N, device=device),
                    ]
                )
            ][:missing]
            valid_vals_b = torch.cat([valid_vals_b, extra])
        elif len(valid_vals_b) > num_rem:
            valid_vals_b = valid_vals_b[:num_rem]

        offspring[b, fill_idx] = valid_vals_b

    return offspring


def calc_broken_pairs_distance(population):
    """
    Computes average Broken Pairs Distance for each individual in the population.
    Distance(A, B) = 1 - (|Edges(A) inter Edges(B)| / N)

    Args:
        population: (B, P, N) tensor of giant tours.
    Returns:
        diversity: (B, P) diversity score (higher is better/more distant).
    """
    B, P, N = population.size()
    device = population.device

    # 1. Construct Edge Hashes
    # Edges: (i, i+1) and (N-1, 0)
    # Hash: min(u,v)*N + max(u,v) (assuming max node index < N? No, nodes are 1..N usually, or indices 0..N-1?)
    # giant_tours usually contains indices.

    # Create cyclic view
    next_nodes = torch.roll(population, shifts=-1, dims=2)

    # Sort u,v to be direction agnostic
    u = torch.min(population, next_nodes)
    v = torch.max(population, next_nodes)

    # Hash (N_max is safely larger than N, e.g. N+1)
    # Be careful if nodes are indices (0..N-1) or IDs (1..N).
    # If 0..N-1, then max hash approx N^2.
    hashes = u * (N + 100) + v  # (B, P, N)

    # 2. Compute Pairwise Distances
    # We want diversity[b, i] = mean_{j != i} (1 - intersection(i, j) / N)
    # Doing full PxP on GPU might be heavy if P is large (e.g. 100).
    # But for P=10-50, B=128, N=100:
    # (B, P, 1, N) == (B, 1, P, N) -> (B, P, P, N) comparison
    # Memory: 128 * 50 * 50 * 100 * 1 byte (bool) = 32 MB. Very Safe.

    # Expand for broadcast
    hashes.unsqueeze(2).unsqueeze(4)  # (B, P, 1, N, 1)
    hashes.unsqueeze(1).unsqueeze(3)  # (B, 1, P, 1, N)

    # Match matrix: matches[b, i, j, k, l]
    intersections = torch.zeros((B, P, P), device=device)
    for i in range(P):
        target = hashes[:, i : i + 1, :]
        matches = hashes.unsqueeze(3) == target.unsqueeze(2)
        num_shared = matches.any(dim=3).sum(dim=2)  # (B, P)
        intersections[:, i, :] = num_shared

    # Distance = 1 - (intersection / N)
    dists = 1.0 - (intersections.float() / N)

    # Diversity of i = mean distance to others
    diversity = dists.sum(dim=2) / max(1, P - 1)
    return diversity


class VectorizedPopulation:
    """
    Manages a population of solutions for the HGS algorithm.

    Handles:
    - Storage of solutions (giant tours), costs, and fitness metrics.
    - Diversity calculation (Broken Pairs Distance).
    - Biased fitness computation (ranking by cost and diversity).
    - Survivor selection.

    Args:
        size (int): Maximum population size.
        device: Torch device.
        alpha_diversity (float): Weight for diversity in biased fitness calculation.
    """

    def __init__(self, size, device, alpha_diversity=0.5):
        """
        Initialize the population.

        Args:
            size (int): Max population size.
            device: Computing device.
            alpha_diversity (float): Diversity weight.
        """
        self.max_size = size
        self.device = device
        self.alpha_diversity = alpha_diversity
        self.population = None  # (B, P, N)
        self.costs = None  # (B, P)
        self.biased_fitness = None  # (B, P)
        self.diversity_scores = None  # (B, P)

    def initialize(self, initial_pop, initial_costs):
        """
        Initializes the population with a set of starting solutions.

        Args:
            initial_pop (torch.Tensor): Initial solutions (giant tours). Shape (B, N) or (B, P0, N).
            initial_costs (torch.Tensor): Costs of initial solutions. Shape (B,) or (B, P0).
        """

        if initial_pop.dim() == 2:
            initial_pop = initial_pop.unsqueeze(1)  # (B, 1, N)

        # Consistent costs shape (B, P0)
        if initial_costs.dim() == 1:
            initial_costs = initial_costs.unsqueeze(1)  # (B, 1)

        self.population = initial_pop
        self.costs = initial_costs
        self.compute_biased_fitness()

    def add_individuals(self, candidates, costs):
        """
        Merges new individuals into the population and selects survivors based on biased fitness.
        Maintains the population size at or below `max_size`.

        Args:
            candidates (torch.Tensor): New solutions to add. Shape (B, C, N).
            costs (torch.Tensor): Costs of new solutions. Shape (B, C).
        """

        if candidates.dim() == 2:
            candidates = candidates.unsqueeze(1)
            costs = costs.unsqueeze(1)

        # 1. Concatenate
        combined_pop = torch.cat([self.population, candidates], dim=1)  # (B, P+C, N)
        combined_costs = torch.cat([self.costs, costs], dim=1)  # (B, P+C)

        # 2. Update state temporarily
        self.population = combined_pop
        self.costs = combined_costs

        # 3. Compute Fitness & Survivor Selection
        self.compute_biased_fitness()

        # Select best P based on biased fitness
        # We want smallest biased_fitness (rank sum)

        if self.population.size(1) > self.max_size:
            # Sort by fitness (ascending, smaller is better)
            _, indices = torch.sort(self.biased_fitness, dim=1)
            survivors = indices[:, : self.max_size]  # (B, max_size)

            # Gather survivors
            B, _, N = self.population.size()

            # Gather population
            surv_expanded = survivors.unsqueeze(2).expand(-1, -1, N)
            self.population = torch.gather(self.population, 1, surv_expanded)

            # Gather costs
            self.costs = torch.gather(self.costs, 1, survivors)

            # Gather fitness
            self.biased_fitness = torch.gather(self.biased_fitness, 1, survivors)

            # Gather diversity (optional)
            if self.diversity_scores is not None:
                self.diversity_scores = torch.gather(self.diversity_scores, 1, survivors)

    def compute_biased_fitness(self):
        """
        Computer Biased Fitness for all individuals in the population.
        Biased Fitness = Rank(Cost) + alpha * Rank(Diversity).
        Lower is better for both ranks (0 is best).
        Updates `self.biased_fitness` and `self.diversity_scores`.
        """

        B, P, N = self.population.size()

        # 1. Rank by Cost (Ascending: lower cost is better, Rank 0)
        cost_indices = torch.argsort(self.costs, dim=1)
        cost_ranks = torch.argsort(cost_indices, dim=1).float()

        # 2. Diversity
        # Calculate diversity (higher is better)
        self.diversity_scores = calc_broken_pairs_distance(self.population)

        # Rank by Diversity (Descending: higher diversity is better, Rank 0)
        div_indices = torch.argsort(self.diversity_scores, dim=1, descending=True)
        div_ranks = torch.argsort(div_indices, dim=1).float()

        # 3. Combine
        # Fitness = CostRank + alpha * DiversityRank
        self.biased_fitness = cost_ranks + self.alpha_diversity * div_ranks

    def get_parents(self, n_offspring=1):
        """
        Selects parents for crossover using binary tournament selection based on biased fitness.

        Args:
            n_offspring (int): Number of offspring to produce (pairs of parents).

        Returns:
            tuple: (parents1, parents2). Both valid tensors of shape (B, n_offspring, N).
        """

        B, P, N = self.population.size()

        def tournament():
            """Selects indices using binary tournament."""
            idx_a = torch.randint(0, P, (B, n_offspring), device=self.device)
            idx_b = torch.randint(0, P, (B, n_offspring), device=self.device)
            fit_a = torch.gather(self.biased_fitness, 1, idx_a)
            fit_b = torch.gather(self.biased_fitness, 1, idx_b)
            return torch.where(fit_a < fit_b, idx_a, idx_b)

        parent1_idx = tournament()
        parent2_idx = tournament()

        p1 = torch.gather(self.population, 1, parent1_idx.unsqueeze(2).expand(-1, -1, N))
        p2 = torch.gather(self.population, 1, parent2_idx.unsqueeze(2).expand(-1, -1, N))

        return p1, p2


class VectorizedHGS:
    """
    Main class for the Vectorized Hybrid Genetic Search (HGS) algorithm.
    Orchestrates the evolution process, including initialization, selection, crossover,
    local search (education), and survivor selection.

    Args:
        dist_matrix (torch.Tensor): Distance matrix.
        demands (torch.Tensor): Demands tensor.
        vehicle_capacity (float/torch.Tensor): Vehicle capacity.
        time_limit (float): Time limit for the search in seconds.
        device: Torch device.
    """

    def __init__(self, dist_matrix, demands, vehicle_capacity, time_limit=1.0, device="cuda"):
        """
        Initialize the HGS solver.
        """
        self.dist_matrix = dist_matrix
        self.demands = demands
        self.vehicle_capacity = vehicle_capacity
        self.time_limit = time_limit
        self.device = device

    def solve(
        self,
        initial_solutions,
        n_generations=50,
        population_size=10,
        elite_size=5,
        time_limit=None,
        max_vehicles=0,
    ):
        """
        Runs the HGS algorithm starting from a set of initial solutions (Expert Imitation Mode).

        Args:
            initial_solutions (torch.Tensor): Initial solutions (giant tours) (B, N).
            n_generations (int): Number of generations to run.
            population_size (int): Size of the genetic population.
            elite_size (int): Number of elite individuals to preserve during restarting.
            time_limit (float, optional): Override time limit in seconds.
            max_vehicles (int, optional): Maximum number of vehicles allowed (0 for unlimited).

        Returns:
            tuple: (best_routes, best_cost)
        """
        B, N = initial_solutions.size()
        start_time = time.time()

        # Initial Evaluation
        _, costs = vectorized_linear_split(
            initial_solutions,
            self.dist_matrix,
            self.demands,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )

        pop = VectorizedPopulation(population_size, self.device)
        pop.initialize(initial_solutions, costs)

        no_improv = 0
        best_cost_tracker = pop.costs.min().item()

        for gen in range(n_generations):
            if time_limit is not None and (time.time() - start_time > time_limit):
                break

            # Selection
            p1, p2 = pop.get_parents(n_offspring=1)
            p1 = p1.squeeze(1)
            p2 = p2.squeeze(1)

            # Crossover
            offspring_giant = vectorized_ordered_crossover(p1, p2)

            # Evaluation
            routes_list, split_costs = vectorized_linear_split(
                offspring_giant,
                self.dist_matrix,
                self.demands,
                self.vehicle_capacity,
                max_vehicles=max_vehicles,
            )

            # Education (Local Search)
            max_l = max(len(r) for r in routes_list)
            offspring_routes = torch.zeros((B, max_l), dtype=torch.long, device=self.device)
            for b in range(B):
                r = routes_list[b]
                offspring_routes[b, : len(r)] = torch.tensor(r, device=self.device)

            if max_l > 2:
                improved_routes = vectorized_two_opt(offspring_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_three_opt(improved_routes, self.dist_matrix, max_iterations=20)
                improved_routes = vectorized_swap(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_relocate(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_two_opt_star(improved_routes, self.dist_matrix, max_iterations=50)
                improved_routes = vectorized_swap_star(improved_routes, self.dist_matrix, max_iterations=50)

                # Recalculate cost
                from_n = improved_routes[:, :-1]
                to_n = improved_routes[:, 1:]

                if self.dist_matrix.dim() == 3 and self.dist_matrix.size(0) == B:
                    batch_ids = torch.arange(B, device=self.device).view(B, 1)
                    dists = self.dist_matrix[batch_ids, from_n, to_n]
                else:
                    expanded_dm = self.dist_matrix
                    if expanded_dm.dim() == 2:
                        expanded_dm = expanded_dm.unsqueeze(0)
                    if expanded_dm.size(0) == 1 and B > 1:
                        expanded_dm = expanded_dm.expand(B, -1, -1)
                    batch_ids = torch.arange(B, device=self.device).view(B, 1)
                    dists = expanded_dm[batch_ids, from_n, to_n]

                improved_costs = dists.sum(dim=1)
            else:
                improved_routes = offspring_routes
                improved_costs = split_costs

            # Reconstruct Giant Tour
            giant_candidates = torch.zeros((B, N), dtype=torch.long, device=self.device)
            improved_routes_cpu = improved_routes.cpu().numpy()

            for b in range(B):
                r_full = improved_routes_cpu[b]
                g_tour = r_full[r_full != 0]
                if len(g_tour) == N:
                    giant_candidates[b, :] = torch.tensor(g_tour, device=self.device)
                else:
                    giant_candidates[b, :] = offspring_giant[b, :]

            # Survivor Selection
            pop.add_individuals(giant_candidates, improved_costs)

            # Stagnation Check
            current_best = pop.costs.min().item()
            if current_best < best_cost_tracker - 1e-4:
                best_cost_tracker = current_best
                no_improv = 0
            else:
                no_improv += 1

            if no_improv > 50:
                k = elite_size
                # Keep Elite
                sorted_idx = torch.argsort(pop.biased_fitness, dim=1)[:, :k]
                elite_pop = torch.gather(pop.population, 1, sorted_idx.unsqueeze(2).expand(-1, -1, N))
                elite_cost = torch.gather(pop.costs, 1, sorted_idx)

                # New candidates (Random Permutations)
                n_rest = pop.max_size - k
                new_pop = torch.zeros((B, n_rest, N), dtype=torch.long, device=self.device)
                for b in range(B):
                    for i in range(n_rest):
                        new_pop[b, i] = torch.randperm(N, device=self.device) + 1

                # Eval new pop
                # Flatten (B, n_rest, N) -> (B*n_rest, N)
                b_sz, n_rst, n_nds = new_pop.shape
                flat_pop = new_pop.view(b_sz * n_rst, n_nds)

                flat_dist = self.dist_matrix.repeat_interleave(n_rst, dim=0)
                flat_demands = self.demands.repeat_interleave(n_rst, dim=0)
                flat_cap = self.vehicle_capacity
                if isinstance(flat_cap, torch.Tensor) and flat_cap.dim() > 0:
                    flat_cap = flat_cap.repeat_interleave(n_rst, dim=0)

                _, new_costs_flat = vectorized_linear_split(
                    flat_pop,
                    flat_dist,
                    flat_demands,
                    flat_cap,
                    max_vehicles=max_vehicles,
                )
                new_costs = new_costs_flat.view(b_sz, n_rst)

                # Update pop directly
                pop.population = torch.cat([elite_pop, new_pop], dim=1)
                pop.costs = torch.cat([elite_cost, new_costs], dim=1)
                pop.compute_biased_fitness()
                no_improv = 0

        best_cost_pop, best_idx = torch.min(pop.costs, dim=1)
        best_giant = pop.population[torch.arange(B), best_idx]
        best_routes, best_final_costs = vectorized_linear_split(
            best_giant,
            self.dist_matrix,
            self.demands,
            self.vehicle_capacity,
            max_vehicles=max_vehicles,
        )
        return best_routes, best_final_costs
