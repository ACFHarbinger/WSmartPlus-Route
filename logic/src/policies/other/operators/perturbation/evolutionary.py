"""
Evolutionary Perturbation Module.

Implements a micro-evolutionary algorithm (rapid GA) applied to a spatial
cluster of routes. Features dynamic route load-balancing via Dummy Depot
Injection and active unvisited node harvesting for VRPP profit maximization.

Algorithm:
1. Select a cluster of routes and flatten them into a "Giant Tour".
2. Inject Dummy Depots (-1) to represent vehicle boundaries.
3. (VRPP Only): Harvest high-profit unvisited nodes and swap them in.
4. Evolve via True OX1 Crossover and Swap Mutation.
5. Decode the optimal split and replace the original cluster.

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

import numpy as np


def evolutionary_perturbation(
    ls: Any,
    target_routes: Optional[List[int]] = None,
    pop_size: int = 10,
    n_generations: int = 5,
    rng: Optional[Random] = None,
) -> bool:
    """
    Micro-evolutionary perturbation for standard CVRP (Cost Minimization).
    Uses Dummy Depot Injection to dynamically resize and balance routes.
    """
    if rng is None:
        rng = Random(42)

    if not ls.routes:
        return False

    if target_routes is None:
        target_routes = _select_target_routes(ls, n=2)

    if not target_routes:
        return False

    # 1. Extract nodes into a flat array
    cluster_nodes: List[int] = []
    for r_idx in target_routes:
        if r_idx < len(ls.routes):
            cluster_nodes.extend(ls.routes[r_idx])

    if len(cluster_nodes) < 3:
        return False

    # 2. Build the baseline chromosome with Dummy Depots (-1)
    base_chromosome = list(cluster_nodes)
    for _ in range(len(target_routes) - 1):
        base_chromosome.append(-1)

    baseline_cost = _eval_cvrp_chromosome(base_chromosome, ls)

    # 3. Generate Population
    population: List[List[int]] = [list(base_chromosome)]
    for _ in range(pop_size - 1):
        individual = list(base_chromosome)
        rng.shuffle(individual)
        population.append(individual)

    # 4. Evolve
    for _ in range(n_generations):
        fitnesses = [_eval_cvrp_chromosome(seq, ls) for seq in population]
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        survivors = [population[i] for i in ranked[: max(2, pop_size // 2)]]

        population = list(survivors)
        while len(population) < pop_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = _order_crossover(p1, p2, rng)
            _mutate_swap(child, rng, prob=0.3)
            population.append(child)

    # 5. Final Evaluation
    fitnesses = [_eval_cvrp_chromosome(seq, ls) for seq in population]
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_seq = population[best_idx]
    best_cost = fitnesses[best_idx]

    if best_cost < baseline_cost - 1e-4:
        _apply_cluster(ls, target_routes, best_seq)
        return True

    return False


def evolutionary_perturbation_profit(  # noqa: C901
    ls: Any,
    target_routes: Optional[List[int]] = None,
    pop_size: int = 10,
    n_generations: int = 5,
    rng: Optional[Random] = None,
) -> bool:
    """
    Micro-evolutionary perturbation for VRPP (Profit Maximization).
    Actively harvests unvisited nodes to escape the Phantom Profit illusion.
    """
    if rng is None:
        rng = Random(42)

    if not ls.routes:
        return False

    if target_routes is None:
        target_routes = _select_target_routes(ls, n=2)

    if not target_routes:
        return False

    # 1. Extract nodes
    cluster_nodes: List[int] = []
    for r_idx in target_routes:
        if r_idx < len(ls.routes):
            cluster_nodes.extend(ls.routes[r_idx])

    if len(cluster_nodes) < 3:
        return False

    # 2. Evaluate Baseline BEFORE injecting new nodes
    base_chromosome_original = list(cluster_nodes)
    for _ in range(len(target_routes) - 1):
        base_chromosome_original.append(-1)

    baseline_profit = _eval_vrpp_chromosome(base_chromosome_original, ls)

    # 3. Active Harvest: Inject high-profit unvisited nodes
    all_visited = {n for r in ls.routes for n in r}
    n_nodes = len(ls.d)
    unvisited = [n for n in range(1, n_nodes) if n not in all_visited]

    if unvisited and cluster_nodes:
        # Perturb up to 3 nodes, or 15% of the cluster
        num_to_swap = max(1, min(3, len(unvisited), len(cluster_nodes) // 6))

        # Sort unvisited by descending profit (Greedy Exploration)
        unvisited_sorted = sorted(unvisited, key=lambda n: _get_waste(ls, n) * getattr(ls, "R", 1.0), reverse=True)
        # Sort cluster by ascending profit (Drop the weakest links)
        cluster_sorted = sorted(cluster_nodes, key=lambda n: _get_waste(ls, n) * getattr(ls, "R", 1.0))

        for i in range(num_to_swap):
            idx_to_replace = cluster_nodes.index(cluster_sorted[i])
            cluster_nodes[idx_to_replace] = unvisited_sorted[i]

    # 4. Build the dynamic chromosome with Dummy Depots
    base_chromosome = list(cluster_nodes)
    for _ in range(len(target_routes) - 1):
        base_chromosome.append(-1)

    # 5. Generate Population
    population: List[List[int]] = [list(base_chromosome)]
    for _ in range(pop_size - 1):
        individual = list(base_chromosome)
        rng.shuffle(individual)
        population.append(individual)

    # 6. Evolve
    for _ in range(n_generations):
        # Fitness is negative profit for rank-sorting
        fitnesses = [-_eval_vrpp_chromosome(seq, ls) for seq in population]
        ranked = sorted(range(len(population)), key=lambda i: fitnesses[i])
        survivors = [population[i] for i in ranked[: max(2, pop_size // 2)]]

        population = list(survivors)
        while len(population) < pop_size:
            p1 = rng.choice(survivors)
            p2 = rng.choice(survivors)
            child = _order_crossover(p1, p2, rng)
            _mutate_swap(child, rng, prob=0.3)
            population.append(child)

    # 7. Final Evaluation
    fitnesses = [-_eval_vrpp_chromosome(seq, ls) for seq in population]
    best_idx = min(range(len(population)), key=lambda i: fitnesses[i])
    best_seq = population[best_idx]
    best_profit = -fitnesses[best_idx]

    # Apples-to-apples comparison against the original baseline
    if best_profit > baseline_profit + 1e-4:
        _apply_cluster(ls, target_routes, best_seq)
        return True

    return False


# --- Core Chromosome Mechanics ---


def _decode_chromosome(seq: List[int]) -> List[List[int]]:
    """Decodes a Giant Tour with -1 dummy depots into a list of route lists."""
    routes: List[List[int]] = []
    current_route: List[int] = []
    for gene in seq:
        if gene == -1:
            routes.append(current_route)
            current_route = []
        else:
            current_route.append(gene)
    routes.append(current_route)
    return routes


def _eval_cvrp_chromosome(seq: List[int], ls: Any) -> float:
    """Evaluates Total Cost, strictly penalizing capacity violations."""
    routes = _decode_chromosome(seq)
    total_cost = 0.0
    penalty = 0.0
    cap = getattr(ls, "capacity", float("inf"))

    for r in routes:
        total_cost += _sequence_cost(ls.d, r)
        load = sum(_get_waste(ls, n) for n in r)
        if load > cap + 1e-6:
            penalty += (load - cap) * 10000.0  # Heavy structural penalty

    return total_cost + penalty


def _eval_vrpp_chromosome(seq: List[int], ls: Any) -> float:
    """Evaluates Total Profit, strictly penalizing capacity violations."""
    routes = _decode_chromosome(seq)
    total_profit = 0.0
    penalty = 0.0
    cap = getattr(ls, "capacity", float("inf"))

    for r in routes:
        total_profit += _sequence_profit(ls, r)
        load = sum(_get_waste(ls, n) for n in r)
        if load > cap + 1e-6:
            penalty += (load - cap) * 10000.0

    return total_profit - penalty


# --- True Order Crossover (OX1) with Dummy Depot Support ---


def _tag_depots(seq: List[int]) -> List[int]:
    """Tags duplicate -1 depots with unique negative IDs (-1, -2, -3...) so OX1 doesn't filter them out."""
    res = []
    depot_id = -1
    for x in seq:
        if x == -1:
            res.append(depot_id)
            depot_id -= 1
        else:
            res.append(x)
    return res


def _untag_depots(seq: List[int]) -> List[int]:
    """Restores unique negative IDs back to the standard -1 dummy depot."""
    return [-1 if x < 0 else x for x in seq]


def _order_crossover(p1: List[int], p2: List[int], rng: Random) -> List[int]:
    """True wrapping Order Crossover (OX1) supporting multiple duplicate routes."""
    n = len(p1)
    if n < 2:
        return list(p1)

    p1_tagged = _tag_depots(p1)
    p2_tagged = _tag_depots(p2)

    c1, c2 = sorted(rng.sample(range(n), 2))
    child = [None] * n
    child[c1:c2] = p1_tagged[c1:c2]  # type: ignore[assignment]

    inherited = set(p1_tagged[c1:c2])
    fill = [g for g in p2_tagged if g not in inherited]

    # Proper OX1 Wrapping
    fill_idx = 0
    for i in range(n):
        target_idx = (c2 + i) % n
        if child[target_idx] is None:
            child[target_idx] = fill[fill_idx]  # type: ignore[call-overload]
            fill_idx += 1

    return _untag_depots(child)  # type: ignore[return-value,arg-type]


def _mutate_swap(seq: List[int], rng: Random, prob: float = 0.3) -> None:
    if len(seq) < 2:
        return
    if rng.random() < prob:
        i, j = rng.sample(range(len(seq)), 2)
        seq[i], seq[j] = seq[j], seq[i]


def _apply_cluster(ls: Any, route_indices: List[int], best_seq: List[int]) -> None:
    """Replaces old rigid routes with the dynamically sized new routes."""
    new_routes = _decode_chromosome(best_seq)
    affected = set()

    for r_idx, new_r in zip(route_indices, new_routes):
        ls.routes[r_idx] = new_r
        affected.add(r_idx)

    if hasattr(ls, "_update_map"):
        ls._update_map(affected)


# --- Utility Functions ---


def _get_waste(ls: Any, node: int) -> float:
    """Safe waste extraction supporting both dicts and arrays."""
    if isinstance(ls.waste, dict):
        return ls.waste.get(node, 0.0)
    return float(ls.waste[node]) if node < len(ls.waste) else 0.0


def _select_target_routes(ls: Any, n: int = 2) -> List[int]:
    route_sizes = [(i, len(r)) for i, r in enumerate(ls.routes) if r]
    route_sizes.sort(key=lambda x: x[1])
    return [i for i, _ in route_sizes[:n]]


def _sequence_cost(d: np.ndarray, seq: List[int]) -> float:
    if not seq:
        return 0.0
    cost = d[0, seq[0]]
    for i in range(len(seq) - 1):
        cost += d[seq[i], seq[i + 1]]
    cost += d[seq[-1], 0]
    return float(cost)


def _sequence_profit(ls: Any, seq: List[int]) -> float:
    if not seq:
        return 0.0
    R = getattr(ls, "R", 1.0)
    C = getattr(ls, "C", 1.0)

    dist = ls.d[0, seq[0]]
    revenue = _get_waste(ls, seq[0]) * R
    for i in range(len(seq) - 1):
        dist += ls.d[seq[i], seq[i + 1]]
        revenue += _get_waste(ls, seq[i + 1]) * R
    dist += ls.d[seq[-1], 0]
    return float(revenue - dist * C)
