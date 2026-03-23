import random
from typing import Optional

from logic.src.policies.hybrid_genetic_search.individual import Individual


def ordered_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
    """
    Variable-Length Ordered Crossover (OX1) for VRPP.

    Safely handles parents that visit different subsets of nodes by dynamically
    sizing the child sequence rather than using a fixed-length array template.

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
        rng = random.Random(42)

    t1 = p1.giant_tour
    t2 = p2.giant_tour

    # Edge case: Empty parents
    if len(t1) < 2 or not t2:
        return Individual(t1[:]) if t1 else Individual(t2[:])

    # 1. Select crossover points from Parent 1
    a, b = sorted(rng.sample(range(len(t1)), 2))

    # 2. Extract the inherited segment
    segment = t1[a : b + 1]
    segment_set = set(segment)

    # 3. Collect remaining nodes from Parent 2 in their exact relative order
    p2_contribution = [node for node in t2 if node not in segment_set]

    # 4. Assemble the child giant tour dynamically
    child_gt = p2_contribution[:a] + segment + p2_contribution[a:]

    return Individual(child_gt)
