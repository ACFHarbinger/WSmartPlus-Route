import random
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

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


def _get_physical_components(adj: Dict[int, List[int]], all_nodes: Set[int]) -> List[List[int]]:
    visited: Set[int] = set([0])  # Mark depot as visited to prevent bridging
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


def route_profit_gpx_crossover(
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
    Route-based Profit Generalized Partition Crossover (RP-GPX).
    Adapts the classical GPX for VRPP by operating on decoded routes
    rather than the giant tour.
    """
    if rng is None:
        rng = random.Random(42)

    if not p1.routes or not p2.routes:
        return Individual(p1.giant_tour[:])

    # 1. Extract physical edges and find intersection
    p1_edges = _get_physical_edges(p1.routes)
    p2_edges = _get_physical_edges(p2.routes)
    common_edges = p1_edges & p2_edges

    # 2. Build adjacency list from common edges
    adj: Dict[int, List[int]] = defaultdict(list)
    for u, v in common_edges:
        if u != 0 and v != 0:
            adj[u].append(v)
            adj[v].append(u)

    all_nodes = set(p1.giant_tour) | set(p2.giant_tour)
    components = _get_physical_components(adj, all_nodes)

    child_routes: List[List[int]] = []
    child_nodes: Set[int] = set()

    # 3. Inherit components as partial or full routes
    for component in components:
        comp_load = sum(wastes.get(n, 0.0) for n in component)
        if comp_load <= capacity:
            parent_tour = p1.giant_tour if rng.random() < 0.5 else p2.giant_tour
            comp_set = set(component)
            ordered_comp = [n for n in parent_tour if n in comp_set]

            if ordered_comp:
                child_routes.append(ordered_comp)
                child_nodes.update(ordered_comp)

    # 4. Pack remaining profitable nodes
    pool = list(all_nodes - child_nodes)
    rng.shuffle(pool)
    loads = [sum(wastes.get(n, 0) for n in r) for r in child_routes]

    for node in pool:
        node_waste = wastes.get(node, 0.0)
        revenue = node_waste * R

        best_profit = -float("inf")
        best_r_idx = -1
        best_pos = -1

        for r_idx, route in enumerate(child_routes):
            if loads[r_idx] + node_waste > capacity:
                continue
            for pos in range(len(route) + 1):
                prev = route[pos - 1] if pos > 0 else 0
                nxt = route[pos] if pos < len(route) else 0

                cost = dist_matrix[prev, node] + dist_matrix[node, nxt] - dist_matrix[prev, nxt]
                profit = revenue - (cost * C)

                if profit > best_profit and profit >= -1e-4:
                    best_profit = profit
                    best_r_idx = r_idx
                    best_pos = pos

        new_cost = dist_matrix[0, node] + dist_matrix[node, 0]
        new_profit = revenue - (new_cost * C)
        seed_hurdle = -0.5 * (new_cost * C)

        if new_profit > best_profit and new_profit >= seed_hurdle:
            child_routes.append([node])
            loads.append(node_waste)
        elif best_r_idx != -1:
            child_routes[best_r_idx].insert(best_pos, node)
            loads[best_r_idx] += node_waste

    # Convert back into giant tour
    child_gt = [node for route in child_routes for node in route]
    ind = Individual(child_gt)
    ind.routes = child_routes
    return ind
