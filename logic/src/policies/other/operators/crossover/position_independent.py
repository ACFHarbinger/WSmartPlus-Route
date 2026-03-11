import random
from typing import Optional

from logic.src.policies.hybrid_genetic_search.individual import Individual


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
        rng = random.Random(42)

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
