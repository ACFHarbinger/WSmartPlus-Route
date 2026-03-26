"""
Evolutionary operators for Hybrid Genetic Search (HGS).

This module contains the crossover, evaluation, and fitness calculation
logic for maintaining and improving the HGS population.
"""

from typing import List

import numpy as np

from .individual import Individual
from .split import LinearSplit


def _extract_edges(individual: Individual) -> set:
    """
    Extract all directed edges from an individual's routes.

    An edge (A, B) exists if node B immediately follows node A in a route,
    including edges from/to the depot (node 0).

    Args:
        individual: Individual whose edges to extract.

    Returns:
        Set of directed edges (tuples of node pairs).
    """
    edges = set()
    for route in individual.routes:
        if not route:
            continue
        # Edge from depot to first node
        edges.add((0, route[0]))
        # Edges within route
        for i in range(len(route) - 1):
            edges.add((route[i], route[i + 1]))
        # Edge from last node back to depot
        edges.add((route[-1], 0))
    return edges


def _compute_broken_pairs_distance(ind1: Individual, ind2: Individual) -> float:
    """
    Compute the broken pairs distance (edge-based diversity) between two individuals.

    Following Vidal (2022), the distance is defined as:
    Δ(A,B) = 1 - |E(A) ∩ E(B)| / max(|E(A)|, |E(B)|)

    Args:
        ind1: First individual.
        ind2: Second individual.

    Returns:
        Broken pairs distance between 0.0 and 1.0.
    """
    edges1 = _extract_edges(ind1)
    edges2 = _extract_edges(ind2)

    if not edges1 and not edges2:
        return 0.0

    if not edges1 or not edges2:
        return 1.0

    intersection = len(edges1 & edges2)
    max_edges = max(len(edges1), len(edges2))

    return 1.0 - (intersection / max_edges)


def update_biased_fitness(population: List[Individual], nb_elite: int, neighbor_size: int = 5):
    """
    Update biased fitness based on profit rank and diversity rank.
    Follows Vidal et al. (2022) HGS-CVRP implementation with parameterless
    diversity weighting.

    Args:
        population: List of individuals to update (single subpopulation).
        nb_elite: Number of elite individuals to protect.
        neighbor_size: Number of nearest neighbors (nbClose) to consider for diversity.
    """
    if not population:
        return
    pop_size = len(population)
    if pop_size == 0:
        return

    # Rank by Profit (or penalized cost for infeasible)
    # For VRPP, higher profit is better
    population.sort(key=lambda x: x.profit_score, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i + 1

    # Diversity: Average broken pairs distance to nbClose closest individuals
    # Based on edge similarity (topological diversity)
    for i, ind1 in enumerate(population):
        dists = []
        for j, ind2 in enumerate(population):
            if i == j:
                continue
            # Compute edge-based distance (broken pairs)
            dist = _compute_broken_pairs_distance(ind1, ind2)
            dists.append(dist)

        # Average distance to nbClose closest individuals
        if dists:
            dists.sort()
            n_neighbors = min(neighbor_size, len(dists))
            ind1.dist_to_parents = float(np.mean(dists[:n_neighbors]))
        else:
            ind1.dist_to_parents = 0.0

    # Rank by Diversity
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    # Parameterless Biased Fitness (Vidal 2022)
    # BF(I) = Rank_C(I) + (1 - N_elite/|Pop|) * Rank_D(I)
    # This automatically balances profit and diversity based on elite proportion
    diversity_weight = 1.0 - (nb_elite / pop_size)

    for ind in population:
        # Apply biased fitness formula
        ind.fitness = ind.rank_profit + diversity_weight * ind.rank_diversity


def evaluate(ind: Individual, split_manager: LinearSplit, penalty_capacity: float = 1.0):
    """
    Decode giant tour and calculate metrics, including feasibility and penalties.

    This function transforms the genotype (giant_tour) into the phenotype (routes)
    using the Split algorithm. For VRPP, the Split algorithm may skip unprofitable nodes,
    resulting in routes that contain a subset of the nodes in giant_tour.

    The giant_tour is preserved unchanged to maintain genetic material for crossover.
    Nodes not in routes are considered "unvisited" and remain available for future
    insertion by local search operators.

    Args:
        ind: Individual to evaluate.
        split_manager: Split algorithm manager.
        penalty_capacity: Penalty coefficient for capacity violations.
    """
    routes, profit = split_manager.split(ind.giant_tour)
    ind.routes = routes
    ind.profit_score = profit

    # Calculate cost/revenue and check feasibility
    rev = 0.0
    cost = 0.0
    total_capacity_violation = 0.0

    for r in routes:
        if not r:
            continue

        # Calculate route distance and revenue
        d = split_manager.dist_matrix[0, r[0]]
        route_load = split_manager.wastes[r[0]]

        for i in range(len(r) - 1):
            d += split_manager.dist_matrix[r[i], r[i + 1]]
            route_load += split_manager.wastes[r[i + 1]]

        d += split_manager.dist_matrix[r[-1], 0]
        cost += d * split_manager.C

        # Calculate revenue for all nodes in route
        for node in r:
            rev += split_manager.wastes[node] * split_manager.R

        # Check capacity constraint
        if route_load > split_manager.capacity:
            total_capacity_violation += route_load - split_manager.capacity

    ind.cost = cost
    ind.revenue = rev
    ind.capacity_violation = total_capacity_violation
    ind.is_feasible = total_capacity_violation == 0.0

    # Penalized cost includes penalty for capacity violations
    ind.penalized_cost = cost + penalty_capacity * total_capacity_violation
