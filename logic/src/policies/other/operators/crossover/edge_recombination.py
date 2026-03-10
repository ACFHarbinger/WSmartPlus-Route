import random
from collections import defaultdict
from typing import Dict, List, Optional, Set

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
        rng = random.Random()

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
