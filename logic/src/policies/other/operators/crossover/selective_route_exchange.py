import random
from typing import Optional

from logic.src.policies.hybrid_genetic_search.individual import Individual

from .ordered import ordered_crossover


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
        rng = random.Random(42)

    if not p1.routes or not p2.routes:
        # Fallback to ordered crossover if routes not available
        return ordered_crossover(p1, p2, rng)

    if not p1.giant_tour:
        return Individual(p2.giant_tour[:])
    if not p2.giant_tour:
        return Individual(p1.giant_tour[:])

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
