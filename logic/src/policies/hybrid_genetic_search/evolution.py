"""
Evolutionary operators for Hybrid Genetic Search (HGS).

This module contains the crossover, evaluation, and fitness calculation
logic for maintaining and improving the HGS population.
"""

import random
from typing import List, Optional

import numpy as np

from .individual import Individual
from .split import LinearSplit


def ordered_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
    """
    Apply Ordered Crossover (OX) to giant tours.
    """
    if rng is None:
        rng = random.Random()
    size = len(p1.giant_tour)
    a, b = sorted(rng.sample(range(size), 2))

    child_gt = [0] * size
    child_gt[a : b + 1] = p1.giant_tour[a : b + 1]

    fill_pos = (b + 1) % size
    source_pos = (b + 1) % size

    p1_set = set(p1.giant_tour[a : b + 1])

    for _ in range(size):
        node = p2.giant_tour[source_pos]
        if node not in p1_set:
            child_gt[fill_pos] = node
            fill_pos = (fill_pos + 1) % size
        source_pos = (source_pos + 1) % size

    return Individual(child_gt)


def update_biased_fitness(population: List[Individual], nb_elite: int, alpha_diversity: float = 0.5):
    """
    Update biased fitness based on profit rank and diversity rank.

    Args:
        population: List of individuals to update.
        nb_elite: Number of elite individuals to protect.
        alpha_diversity: Weight for diversity in fitness calculation (0.0 = pure profit, 1.0 = pure diversity).
    """
    if not population:
        return
    pop_size = len(population)
    if pop_size == 0:
        return

    # Rank by Profit
    population.sort(key=lambda x: x.profit_score, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_profit = i + 1

    # Diversity: Average distance to closest individuals based on VISITED node sets
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
        dists.sort()
        # Avg of closest 5
        ind1.dist_to_parents = float(np.mean(dists[:5])) if dists else 0.0

    # Rank by Diversity
    population.sort(key=lambda x: x.dist_to_parents, reverse=True)
    for i, ind in enumerate(population):
        ind.rank_diversity = i + 1

    # Biased Fitness = Rank(Profit) + alpha_diversity * Rank(Diversity)
    # ELITE PROTECTION: Ensure top nb_elite profit solutions always survive.
    # Their fitness is bounded by [1, nb_elite], while others are >= nb_elite + 1.
    for ind in population:
        if ind.rank_profit <= nb_elite:
            # Pure profit ranking for top elites ensures their survival in trim
            ind.fitness = float(ind.rank_profit)
        else:
            # Biased fitness for the rest to maintain diversity
            ind.fitness = ind.rank_profit + alpha_diversity * ind.rank_diversity


def evaluate(ind: Individual, split_manager: LinearSplit):
    """
    Decode giant tour and calculate metrics.
    """
    routes, profit = split_manager.split(ind.giant_tour)
    ind.routes = routes
    ind.profit_score = profit

    # Calculate cost/rev
    rev = 0.0
    cost = 0.0
    for r in routes:
        if not r:
            continue
        d = split_manager.dist_matrix[0, r[0]]
        rev += split_manager.wastes[r[0]] * split_manager.R
        for i in range(len(r) - 1):
            d += split_manager.dist_matrix[r[i], r[i + 1]]
            rev += split_manager.wastes[r[i + 1]] * split_manager.R
        d += split_manager.dist_matrix[r[-1], 0]
        cost += d * split_manager.C

    ind.cost = cost
    ind.revenue = rev
