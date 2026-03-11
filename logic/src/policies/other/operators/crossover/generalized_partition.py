import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

from logic.src.policies.hybrid_genetic_search.individual import Individual


def get_edges(tour: List[int]) -> Set[Tuple[int, int]]:
    """
    Build edge sets from both parents (including depot connections).
    """
    edges: Set[Tuple[int, int]] = set()
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


def get_components(adj: Dict[int, List[int]], all_nodes: Set[int]) -> List[List[int]]:
    """
    Find connected components using Depth First Search.
    """
    visited: Set[int] = set([0])  # Mark depot as visited
    components: List[List[int]] = []

    def dfs(node: int, component: List[int]):
        visited.add(node)
        component.append(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in all_nodes:
        if node not in visited and node != 0:
            component: List[int] = []
            dfs(node, component)
            if component:
                components.append(component)
    return components


def generalized_partition_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
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
        rng = random.Random(42)

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
    components = get_components(adj, all_nodes)

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
    # Preservation of p1's structure for remaining
    for node in p1.giant_tour:
        if node not in child_set and node != 0:
            child_gt.append(node)

    # Secondary check from p2
    child_set = set(child_gt)
    for node in p2.giant_tour:
        if node not in child_set and node != 0:
            child_gt.append(node)

    return Individual(child_gt)
