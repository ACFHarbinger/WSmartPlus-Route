import random
from typing import Optional

from logic.src.policies.hybrid_genetic_search.individual import Individual


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

    size1 = len(p1.giant_tour)
    size2 = len(p2.giant_tour)

    if size1 < 2 or size2 == 0:
        return Individual(p1.giant_tour[:])

    # Select crossover points
    a, b = sorted(rng.sample(range(size1), 2))

    # Initialize child with zeros
    child_gt = [0] * size1
    child_gt[a : b + 1] = p1.giant_tour[a : b + 1]

    # Fill remaining positions from parent 2
    fill_pos = (b + 1) % size1
    source_pos = (b + 1) % size2
    p1_set = set(p1.giant_tour[a : b + 1])

    for _ in range(size2):
        node = p2.giant_tour[source_pos]
        if node not in p1_set and fill_pos < size1:
            child_gt[fill_pos] = node
            fill_pos = (fill_pos + 1) % size1
        source_pos = (source_pos + 1) % size2

    # If child is not full (size1 > size2 or excessive overlaps), fill with missing nodes
    if 0 in child_gt:
        visited = set(child_gt)
        if 0 in visited:
            visited.remove(0)
        missing = [n for n in p1.giant_tour if n not in visited]
        for i in range(size1):
            if child_gt[i] == 0 and missing:
                child_gt[i] = missing.pop(0)

    return Individual(child_gt)
