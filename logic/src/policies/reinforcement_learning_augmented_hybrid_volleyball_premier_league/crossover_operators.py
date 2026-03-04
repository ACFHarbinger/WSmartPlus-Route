"""
Crossover Operators for Hybrid Genetic Search.

This module implements five advanced crossover operators for VRP genetic algorithms:

1. **Ordered Crossover (OX)** - Davis, 1985
   - Preserves relative order of elements
   - Good for TSP-like problems

2. **Position Independent Crossover (PIX)** - Adapted for VRP
   - Focuses on which nodes to inherit, not their positions
   - Excellent for node selection in VRPP

3. **Selective Route Exchange Crossover (SREX)** - Custom
   - Exchanges complete routes between parents
   - Preserves good route structures

4. **Generalized Partition Crossover (GPX)** - Whitley et al., 2009
   - Graph-based recombination using common edges
   - Creates offspring from edge unions

5. **Edge Recombination Crossover (ERX)** - Whitley, 1989
   - Preserves edges from parents
   - Builds offspring by following edge adjacencies

Reference:
    Vidal et al., "A hybrid genetic algorithm for multidepot and periodic VRP", 2012.
    Nagata & Bräysy, "Edge assembly-based memetic algorithm", 2009.
"""

import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from .individual import Individual


def ordered_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
    """
    Ordered Crossover (OX): Preserves relative order from parents.

    Algorithm:
        1. Select random segment from parent 1
        2. Copy segment to offspring
        3. Fill remaining positions with parent 2's order, skipping duplicates

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    size = len(p1.giant_tour)
    if size == 0:
        return Individual([])

    # Select crossover points
    a, b = sorted(rng.sample(range(size), 2))

    # Initialize child with zeros
    child_gt = [0] * size
    child_gt[a : b + 1] = p1.giant_tour[a : b + 1]

    # Fill remaining positions from parent 2
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


def position_independent_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
    """
    Position Independent Crossover (PIX): Focuses on node inheritance.

    Algorithm:
        1. For each node, randomly choose parent to inherit from
        2. Build offspring using inherited nodes in parent 1's order
        3. Add non-inherited nodes from parent 2's order

    This is particularly good for VRPP where node selection matters more than order.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    # Get all unique nodes from both parents
    all_nodes = set(p1.giant_tour) | set(p2.giant_tour)

    # Randomly assign each node to a parent
    from_p1 = set()
    from_p2 = set()

    for node in all_nodes:
        if rng.random() < 0.5:
            from_p1.add(node)
        else:
            from_p2.add(node)

    # Build child: first add nodes from p1 in p1's order
    child_gt = []
    for node in p1.giant_tour:
        if node in from_p1:
            child_gt.append(node)
            from_p1.discard(node)

    # Then add nodes from p2 in p2's order
    for node in p2.giant_tour:
        if node in from_p2:
            child_gt.append(node)
            from_p2.discard(node)

    # Add any remaining nodes (shouldn't happen, but safety check)
    for node in from_p1:
        child_gt.append(node)
    for node in from_p2:
        child_gt.append(node)

    return Individual(child_gt)


def selective_route_exchange_crossover(
    p1: Individual, p2: Individual, rng: Optional[random.Random] = None
) -> Individual:
    """
    Selective Route Exchange Crossover (SREX): Exchanges complete routes.

    Algorithm:
        1. Randomly select subset of routes from parent 1
        2. Add non-conflicting routes from parent 2
        3. Add remaining nodes in original order

    Preserves good route structures from both parents.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    if not p1.routes or not p2.routes:
        # Fallback to ordered crossover if routes not available
        return ordered_crossover(p1, p2, rng)

    # Select random routes from parent 1
    n_routes_p1 = max(1, len(p1.routes) // 2)
    selected_p1_routes = rng.sample(p1.routes, min(n_routes_p1, len(p1.routes)))

    # Collect nodes from selected routes
    child_nodes = set()
    child_routes = []
    for route in selected_p1_routes:
        child_routes.append(route[:])
        child_nodes.update(route)

    # Add non-conflicting routes from parent 2
    for route in p2.routes:
        route_nodes = set(route)
        if not route_nodes & child_nodes:  # No overlap
            child_routes.append(route[:])
            child_nodes.update(route)

    # Convert routes to giant tour
    child_gt = []
    for route in child_routes:
        child_gt.extend(route)

    # Add missing nodes from p1's order
    for node in p1.giant_tour:
        if node not in child_nodes:
            child_gt.append(node)
            child_nodes.add(node)

    # Add any remaining missing nodes from p2
    for node in p2.giant_tour:
        if node not in child_nodes:
            child_gt.append(node)
            child_nodes.add(node)

    return Individual(child_gt)


def generalized_partition_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:  # noqa: C901
    """
    Generalized Partition Crossover (GPX): Graph-based recombination.

    Algorithm:
        1. Build union graph of edges from both parents
        2. Identify common edges (present in both parents)
        3. Partition graph into connected components
        4. Recombine components to create offspring

    Preserves common edge structures shared by parents.

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.

    Returns:
        Child individual.
    """
    if rng is None:
        rng = random.Random()

    # Build edge sets from both parents (including depot connections)
    def get_edges(tour: List[int]) -> Set[Tuple[int, int]]:
        edges = set()
        if not tour:
            return edges
        # Depot to first node
        edges.add((0, tour[0]))
        # Tour edges
        for i in range(len(tour) - 1):
            edges.add((tour[i], tour[i + 1]))
        # Last node to depot
        edges.add((tour[-1], 0))
        return edges

    p1_edges = get_edges(p1.giant_tour)
    p2_edges = get_edges(p2.giant_tour)

    # Find common edges
    common_edges = p1_edges & p2_edges

    # Build adjacency list from common edges
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in common_edges:
        if u != 0 and v != 0:  # Exclude depot for partitioning
            adj[u].append(v)
            adj[v].append(u)

    # Find connected components using DFS
    all_nodes = set(p1.giant_tour) | set(p2.giant_tour)
    visited = set([0])  # Mark depot as visited
    components = []

    def dfs(node: int, component: List[int]):
        visited.add(node)
        component.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in all_nodes:
        if node not in visited and node != 0:
            component = []
            dfs(node, component)
            if component:
                components.append(component)

    # Randomly select parent to determine component order
    if rng.random() < 0.5:
        # Use p1's order within each component
        child_gt = []
        for component in components:
            component_set = set(component)
            for node in p1.giant_tour:
                if node in component_set:
                    child_gt.append(node)
    else:
        # Use p2's order within each component
        child_gt = []
        for component in components:
            component_set = set(component)
            for node in p2.giant_tour:
                if node in component_set:
                    child_gt.append(node)

    # Add any missing nodes
    child_set = set(child_gt)
    for node in p1.giant_tour:
        if node not in child_set and node != 0:
            child_gt.append(node)

    return Individual(child_gt)


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


# Crossover operator registry
CROSSOVER_OPERATORS = {
    "OX": ordered_crossover,
    "PIX": position_independent_crossover,
    "SREX": selective_route_exchange_crossover,
    "GPX": generalized_partition_crossover,
    "ERX": edge_recombination_crossover,
}

CROSSOVER_NAMES = list(CROSSOVER_OPERATORS.keys())
