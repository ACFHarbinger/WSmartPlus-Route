import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from logic.src.policies.hybrid_genetic_search.individual import Individual


def edge_recombination_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
    """
    Edge Recombination Crossover (ERX): Preserves edge adjacencies.

    Algorithm:
        1. Build edge adjacency table from both parents
        2. Start from random node
        3. Iteratively select next node with fewest remaining edges
        4. Update adjacency table after each selection

    Maximizes preservation of parent edges in offspring.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random(42)

    # Build edge adjacency table
    adj_table: Dict[int, Set[int]] = defaultdict(set)

    def add_tour_edges(tour: List[int]):
        for i in range(len(tour)):
            current = tour[i]
            # Add neighbors (predecessor and successor)
            prev_node = tour[i - 1] if i > 0 else tour[-1]
            next_node = tour[i + 1] if i < len(tour) - 1 else tour[0]

            if current != 0:  # Exclude depot
                if prev_node != 0:
                    adj_table[current].add(prev_node)
                if next_node != 0:
                    adj_table[current].add(next_node)

    add_tour_edges(p1.giant_tour)
    add_tour_edges(p2.giant_tour)

    # Start from random node
    all_nodes = list(set(p1.giant_tour) | set(p2.giant_tour))
    all_nodes = [n for n in all_nodes if n != 0]  # Exclude depot

    if not all_nodes:
        return Individual([])

    child_gt = []
    current = rng.choice(all_nodes)
    child_gt.append(current)
    remaining = set(all_nodes)
    remaining.remove(current)

    # Build tour by selecting nodes with fewest edges
    while remaining:
        # Get neighbors of current node
        neighbors = adj_table[current] & remaining

        if neighbors:
            # Select neighbor with fewest edges (ties broken randomly)
            next_node = min(
                neighbors,
                key=lambda n: (len(adj_table[n] & remaining), rng.random()),
            )
        else:
            # No neighbors available, select random remaining node
            next_node = rng.choice(list(remaining))

        child_gt.append(next_node)
        remaining.remove(next_node)

        # Remove selected node from all adjacency lists
        for node in remaining:
            adj_table[node].discard(next_node)

        current = next_node

    return Individual(child_gt)


def _get_physical_edges(routes: List[List[int]]) -> Set[Tuple[int, int]]:
    edges: Set[Tuple[int, int]] = set()
    for route in routes:
        if not route:
            continue
        edges.add((0, route[0]))
        for i in range(len(route) - 1):
            edges.add((route[i], route[i + 1]))
        edges.add((route[-1], 0))
    return edges


def capacity_aware_erx(
    p1: Individual,
    p2: Individual,
    dist_matrix: np.ndarray,
    wastes: Dict[int, float],
    capacity: float,
    R: float = 1.0,
    C: float = 1.0,
    rng: Optional[random.Random] = None,
) -> Individual:
    """
    Capacity-Aware Edge Recombination Crossover (C-ERX).
    Adapts the classical ERX for VRPP by walking the physical adjacency matrix of
    both parents while strictly enforcing vehicle capacity and using profit tie-breakers.
    """
    if rng is None:
        rng = random.Random(42)

    if not p1.routes or not p2.routes:
        return Individual(p1.giant_tour[:])

    # 1. Build Union Adjacency from physical routes
    all_edges = _get_physical_edges(p1.routes) | _get_physical_edges(p2.routes)
    adj_table: Dict[int, Set[int]] = defaultdict(set)

    for u, v in all_edges:
        if u != 0 and v != 0:
            adj_table[u].add(v)
            adj_table[v].add(u)

    all_nodes = list(set(p1.giant_tour) | set(p2.giant_tour))
    if not all_nodes:
        return Individual([])

    child_routes: List[List[int]] = []
    current_route: List[int] = []
    current_load = 0.0
    remaining = set(all_nodes)

    # Start at a random node
    current_node = rng.choice(all_nodes)
    current_route.append(current_node)
    current_load += wastes.get(current_node, 0.0)
    remaining.remove(current_node)

    # 2. The Capacity Walk
    while remaining:
        raw_neighbors = adj_table[current_node] & remaining
        valid_neighbors = [n for n in raw_neighbors if current_load + wastes.get(n, 0.0) <= capacity]

        if valid_neighbors:
            # 3. Profit Tie-Breaker
            next_node = min(
                valid_neighbors,
                key=lambda n: (
                    len(adj_table[n] & remaining),
                    -(wastes.get(n, 0.0) * R - dist_matrix[current_node, n] * C),
                ),
            )
            current_route.append(next_node)
            current_load += wastes.get(next_node, 0.0)
            remaining.remove(next_node)
            current_node = next_node

        else:
            # Capacity full or dead end -> Return to depot
            child_routes.append(current_route)
            current_route = []
            current_load = 0.0

            current_node = rng.choice(list(remaining))
            current_route.append(current_node)
            current_load += wastes.get(current_node, 0.0)
            remaining.remove(current_node)

    if current_route:
        child_routes.append(current_route)

    # Convert evaluated routes back into a giant tour
    child_gt = [node for route in child_routes for node in route]

    ind = Individual(child_gt)
    ind.routes = child_routes
    return ind
