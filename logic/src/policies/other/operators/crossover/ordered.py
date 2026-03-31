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
        rng = random.Random()

    t1 = p1.giant_tour
    t2 = p2.giant_tour
    n = len(t1)

    # Edge case: Empty or small parents
    if n < 2 or len(t2) < 2:
        return Individual(t1[:] if t1 else t2[:], expand_pool=p1.expand_pool)

    # Strict consistency checks
    assert len(t1) == len(t2), f"Parent tours must have equal length: {len(t1)} vs {len(t2)}"
    assert p1.expand_pool == p2.expand_pool, "Parents must have matching expand_pool settings"

    # 1. Select crossover points
    a, b = sorted(rng.sample(range(n), 2))

    # 2. Initialize child and copy segment from p1 into exact same positions
    child_tour = [-1] * n
    child_tour[a : b + 1] = t1[a : b + 1]
    segment_set = set(t1[a : b + 1])

    # 3. Circularly sweep p2 to fill the remaining positions
    curr_p2 = (b + 1) % n
    curr_child = (b + 1) % n

    # O(n) fill logic with safety break for mismatched node pools
    total_to_fill = n - (b - a + 1)
    filled = 0
    while filled < total_to_fill:
        node = t2[curr_p2]
        if node not in segment_set:
            child_tour[curr_child] = node
            curr_child = (curr_child + 1) % n
            filled += 1

        curr_p2 = (curr_p2 + 1) % n
        if curr_p2 == (b + 1) % n and filled < total_to_fill:
            break  # Guard against mismatched node pools

    # Final fallback for incomplete tours (should not happen in valid HGS)
    if -1 in child_tour:
        # Fill remaining slots with any missing nodes from a canonical set if necessary
        # For now, just copy from t1 to maintain length and valid type
        for i in range(n):
            if child_tour[i] == -1:
                child_tour[i] = t1[i]

    return Individual(giant_tour=child_tour, expand_pool=p1.expand_pool)
