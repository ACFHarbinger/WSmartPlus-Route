"""
Evolutionary operators for Hybrid Genetic Search (HGS).

This module contains the crossover, evaluation, and fitness calculation
logic for maintaining and improving the HGS population.
"""

from typing import List

import numpy as np

from .individual import Individual
from .split import LinearSplit


def update_biased_fitness(
    population: List[Individual], nb_elite: int, alpha_diversity: float = 0.5, neighbor_size: int = 5
):
    """
    Update biased fitness based on profit rank and diversity rank.
    Follows Vidal et al. (2022) HGS-CVRP implementation.

    Args:
        population: List of individuals to update (single subpopulation).
        nb_elite: Number of elite individuals to protect.
        alpha_diversity: Weight for diversity in fitness calculation (0.0 = pure profit, 1.0 = pure diversity).
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

    # Diversity: Average distance to nbClose closest individuals
    # Based on Hamming distance of visited node sets
    for i, ind1 in enumerate(population):
        dists = []
        s1 = set(n for r in ind1.routes for n in r)
        for j, ind2 in enumerate(population):
            if i == j:
                continue
            s2 = set(n for r in ind2.routes for n in r)
            # Hamming-like distance on set of visited nodes
            dist = len(s1.symmetric_difference(s2))
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

    # Biased Fitness = Rank(Profit) + alpha_diversity * Rank(Diversity)
    # ELITE PROTECTION: Ensure top nb_elite profit solutions always survive
    for ind in population:
        if ind.rank_profit <= nb_elite:
            # Pure profit ranking for elite ensures survival
            ind.fitness = float(ind.rank_profit)
        else:
            # Biased fitness for the rest to maintain diversity
            ind.fitness = ind.rank_profit + alpha_diversity * ind.rank_diversity


def evaluate(ind: Individual, split_manager: LinearSplit, penalty_capacity: float = 1.0):
    """
    Decode giant tour and calculate metrics, including feasibility and penalties.

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
