import random
from typing import Optional

from logic.src.policies.hybrid_genetic_search.individual import Individual


def ordered_crossover(p1: Individual, p2: Individual, rng: Optional[random.Random] = None) -> Individual:
    """
    Classical Ordered Crossover (OX1) for fixed-length permutations.

    Preserves absolute positioning of the segment from Parent 1, and
    circular relative positioning of the remaining nodes from Parent 2.
    Requires giant_tour to contain all nodes (visited + unvisited).

    Algorithm:
        1. Select random segment [a, b] from Parent 1
        2. Copy segment into child at EXACT same positions [a, b]
        3. Fill remaining positions by circularly sweeping Parent 2 from position (b+1)
        4. Skip nodes already in the segment

    This maintains:
        - Absolute positioning inheritance from Parent 1 (spatial traits)
        - Relative ordering inheritance from Parent 2 (sequential traits)
        - Fixed-length genotype for all nodes (visited + unvisited)

    Args:
        p1: First parent individual.
        p2: Second parent individual.
        rng: Random number generator.

    Returns:
        Child individual with fixed-length giant tour.
    """
    if rng is None:
        rng = random.Random(42)

    t1 = p1.giant_tour
    t2 = p2.giant_tour
    n = len(t1)

    # Edge case: Empty or invalid parents
    if n < 2 or not t2:
        return Individual(t1[:], expand_pool=p1.expand_pool) if t1 else Individual(t2[:], expand_pool=p2.expand_pool)

    # 1. Select crossover points
    a, b = sorted(rng.sample(range(n), 2))

    # 2. Initialize child and copy segment from p1 into exact same positions
    child_tour = [-1] * n
    child_tour[a : b + 1] = t1[a : b + 1]
    segment_set = set(t1[a : b + 1])

    # 3. Circularly sweep p2 to fill the remaining positions
    curr_p2 = (b + 1) % n
    curr_child = (b + 1) % n

    # Fill until there are no placeholders left
    while -1 in child_tour:
        node = t2[curr_p2]
        if node not in segment_set:
            child_tour[curr_child] = node
            curr_child = (curr_child + 1) % n
        curr_p2 = (curr_p2 + 1) % n

    return Individual(giant_tour=child_tour, expand_pool=p1.expand_pool)
