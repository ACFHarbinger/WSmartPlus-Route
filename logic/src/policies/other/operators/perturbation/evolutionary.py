"""
Evolutionary Perturbation Module.

Implements a micro-evolutionary algorithm (rapid GA) applied to a spatial
cluster of routes within the solution.  This provides intense, localised
diversification without perturbing the entire solution.

Algorithm:
1. Select a cluster of ``target_routes`` from the solution.
2. Create a small population by applying random perturbations.
3. Run a brief evolutionary loop (crossover + local search) for a few generations.
4. Replace the original cluster with the best individual found.

Attributes:
    None

Example:
    >>> from logic.src.policies.other.operators.perturbation.evolutionary import (
    ...     evolutionary_perturbation,
    ... )
    >>> improved = evolutionary_perturbation(ls, target_routes=[0, 1], pop_size=10, n_gen=5)
"""

from random import Random
from typing import Any, List, Optional


def evolutionary_perturbation(
    ls: Any,
    target_routes: Optional[List[int]] = None,
    pop_size: int = 10,
    n_generations: int = 5,
    rng: Optional[Random] = None,
) -> bool:
    """
    Micro-evolutionary perturbation on a subset of routes.

    Creates a small population of route variants, evolves them briefly
    via order-crossover and random-swap mutation, and replaces the
    original routes if the best individual improves over the baseline.

    Args:
        ls: LocalSearch instance.
        target_routes: Indices of routes to evolve (default: 2 smallest).
        pop_size: Population size for the micro-GA.
        n_generations: Number of evolutionary generations.
        rng: Random number generator.

    Returns:
        bool: True if the evolved cluster improves over the original.
    """
    if rng is None:
        rng = Random(42)

    if not ls.routes:
        return False

    if target_routes is None:
        target_routes = _select_target_routes(ls, n=2)

    if not target_routes:
        return False

    # 1. Extract nodes and RECORD original route sizes for decoding
    cluster_nodes: List[int] = []
    original_sizes: List[int] = []
    for r_idx in target_routes:
        if r_idx < len(ls.routes):
            route = ls.routes[r_idx]
            cluster_nodes.extend(route)
            original_sizes.append(len(route))

    if len(cluster_nodes) < 3:
        return False

    # Baseline cost must be multi-route
    baseline_cost = _cluster_cost(ls, target_routes)

    # Generate population
    population: List[List[int]] = [list(cluster_nodes)]
    for _ in range(pop_size - 1):
        individual = list(cluster_nodes)
        rng.shuffle(individual)
        population.append(individual)

    # 2. Helper: Decode chromosome into multi-route cost
    def _eval_chromosome_cost(seq: List[int]) -> float:
        total_cost = 0.0
        pos = 0
        for size in original_sizes:
            sub_route = seq[pos : pos + size]
            # Sum cost of each sub-route including depot returns
            total_cost += _sequence_cost(ls.d, sub_route)
            pos += size
        return total_cost

    # Evolve
    for _ in range(n_generations):
        # Evaluate fitness using the multi-route decoder
        fitnesses = [_eval_chromosome_cost(seq) for seq in population]

        # Select top half
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        survivors = [population[i] for i in ranked[: max(2, pop_size // 2)]]

        # Create offspring
        population = list(survivors)
        while len(population) < pop_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = _order_crossover(p1, p2, rng)
            _mutate_swap(child, rng, prob=0.3)
            population.append(child)

    # 3. Final evaluation
    fitnesses = [_eval_chromosome_cost(seq) for seq in population]
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_seq = population[best_idx]
    best_cost = fitnesses[best_idx]

    # Apples-to-apples comparison of multi-route costs
    if best_cost < baseline_cost - 1e-4:
        _apply_cluster(ls, target_routes, best_seq)
        return True

    return False


def evolutionary_perturbation_profit(
    ls: Any,
    target_routes: Optional[List[int]] = None,
    pop_size: int = 10,
    n_generations: int = 5,
    rng: Optional[Random] = None,
) -> bool:
    """
    Micro-evolutionary perturbation for profit-maximization.

    Similar to evolutionary_perturbation but uses profit (Revenue - Cost)
    as the objective function.

    Args:
        ls: LocalSearch instance.
        target_routes: Indices of routes to evolve.
        pop_size: Population size.
        n_generations: Number of generations.
        rng: Random number generator.

    Returns:
        bool: True if the evolved cluster improves over the original.
    """
    if rng is None:
        rng = Random(42)

    if not ls.routes:
        return False

    if target_routes is None:
        target_routes = _select_target_routes(ls, n=2)

    if not target_routes:
        return False

    # 1. Extract the target cluster and record original route sizes!
    cluster_nodes: List[int] = []
    original_sizes: List[int] = []
    for r_idx in target_routes:
        if r_idx < len(ls.routes):
            route = ls.routes[r_idx]
            cluster_nodes.extend(route)
            original_sizes.append(len(route))

    if len(cluster_nodes) < 3:
        return False

    baseline_profit = _cluster_profit(ls, target_routes)

    # Generate population of permutations
    population: List[List[int]] = [list(cluster_nodes)]
    for _ in range(pop_size - 1):
        individual = list(cluster_nodes)
        rng.shuffle(individual)
        population.append(individual)

    # 2. Helper function to decode and evaluate chromosome correctly
    def _eval_chromosome(seq: List[int]) -> float:
        total_profit = 0.0
        pos = 0
        for size in original_sizes:
            sub_route = seq[pos : pos + size]
            total_profit += _sequence_profit(ls, sub_route)
            pos += size
        return total_profit

    # Evolve
    for _ in range(n_generations):
        # Evaluate fitness (negative profit for rank-sorting)
        fitnesses = [-_eval_chromosome(seq) for seq in population]

        # Select top half
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        survivors = [population[i] for i in ranked[: max(2, pop_size // 2)]]

        # Create offspring
        population = list(survivors)
        while len(population) < pop_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = _order_crossover(p1, p2, rng)
            _mutate_swap(child, rng, prob=0.3)
            population.append(child)

    # 3. Find best individual using the correct evaluator
    fitnesses = [-_eval_chromosome(seq) for seq in population]
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_seq = population[best_idx]
    best_profit = -fitnesses[best_idx]

    # Now this is an apples-to-apples comparison
    if best_profit > baseline_profit + 1e-4:
        _apply_cluster(ls, target_routes, best_seq)
        return True

    return False


def _cluster_profit(ls: Any, route_indices: List[int]) -> float:
    """Compute total profit of selected routes."""
    total_profit = 0.0
    for r_idx in route_indices:
        if r_idx < len(ls.routes):
            total_profit += _sequence_profit(ls, ls.routes[r_idx])
    return total_profit


def _sequence_profit(ls: Any, seq: List[int]) -> float:
    """Total profit of depot → seq → depot."""
    if not seq:
        return 0.0
    dist = ls.d[0, seq[0]]
    revenue = ls.waste.get(seq[0], 0) * ls.R
    for i in range(len(seq) - 1):
        dist += ls.d[seq[i], seq[i + 1]]
        revenue += ls.waste.get(seq[i + 1], 0) * ls.R
    dist += ls.d[seq[-1], 0]
    return float(revenue - dist * ls.C)


def _select_target_routes(ls: Any, n: int = 2) -> List[int]:
    """Select the n smallest non-empty routes as targets."""
    route_sizes = [(i, len(r)) for i, r in enumerate(ls.routes) if r]
    route_sizes.sort(key=lambda x: x[1])
    return [i for i, _ in route_sizes[:n]]


def _cluster_cost(ls: Any, route_indices: List[int]) -> float:
    """Compute total cost of depot→route→depot for selected routes."""
    cost = 0.0
    for r_idx in route_indices:
        if r_idx < len(ls.routes):
            cost += _sequence_cost(ls.d, ls.routes[r_idx])
    return cost


def _sequence_cost(d, seq: List[int]) -> float:
    """Total edge cost of depot → seq → depot."""
    if not seq:
        return 0.0
    cost = d[0, seq[0]]
    for i in range(len(seq) - 1):
        cost += d[seq[i], seq[i + 1]]
    cost += d[seq[-1], 0]
    return float(cost)


def _order_crossover(p1: List[int], p2: List[int], rng: Random) -> List[int]:
    """Order crossover (OX1) between two permutations."""
    n = len(p1)
    if n < 2:
        return list(p1)
    c1, c2 = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[c1:c2] = p1[c1:c2]
    inherited = set(p1[c1:c2])
    fill = [g for g in p2 if g not in inherited]
    idx = 0
    for i in range(n):
        if child[i] is None:
            child[i] = fill[idx]
            idx += 1
    return child  # type: ignore[return-value]


def _mutate_swap(seq: List[int], rng: Random, prob: float = 0.3) -> None:
    """Random swap mutation."""
    if len(seq) < 2:
        return
    if rng.random() < prob:
        i, j = rng.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]


def _apply_cluster(ls: Any, route_indices: List[int], best_seq: List[int]) -> None:
    """Partition best sequence back into routes with original sizes."""
    sizes = [len(ls.routes[i]) for i in route_indices]
    pos = 0
    affected = set()
    for r_idx, size in zip(route_indices, sizes):
        ls.routes[r_idx] = best_seq[pos : pos + size]
        pos += size
        affected.add(r_idx)
    # Handle leftover nodes (if sizes changed)
    if pos < len(best_seq) and route_indices:
        ls.routes[route_indices[-1]].extend(best_seq[pos:])
    ls._update_map(affected)
