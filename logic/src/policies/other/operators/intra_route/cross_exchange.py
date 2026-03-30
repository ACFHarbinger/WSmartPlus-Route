import copy
import random
from typing import List


def apply_intra_route_cross_exchange(routes: List[List[int]], rng: random.Random) -> List[List[int]]:
    """
    Applies the Intra-Route CROSS-exchange operator.
    Randomly selects a route, selects two non-overlapping segments, and swaps them.
    Segment lengths are chosen randomly between 1 and 3.
    """
    valid_routes = [i for i, r in enumerate(routes) if len(r) >= 4]

    if not valid_routes:
        return copy.deepcopy(routes)

    new_routes = copy.deepcopy(routes)
    route_idx = rng.choice(valid_routes)
    route = new_routes[route_idx]
    n = len(route)

    valid_indices = False
    attempts = 0
    indices = []

    while not valid_indices and attempts < 100:
        indices = sorted(rng.sample(range(n + 1), 4))
        i1, j1, i2, j2 = indices
        if 1 <= j1 - i1 <= 3 and 1 <= j2 - i2 <= 3:
            valid_indices = True
        attempts += 1

    if not valid_indices:
        # Fallback if we somehow fail to find lengths 1-3
        indices = sorted(rng.sample(range(n + 1), 4))

    i1, j1, i2, j2 = indices

    seg_a = route[i1:j1]
    seg_b = route[i2:j2]

    # Swap seg_a and seg_b inside the route
    new_route = route[:i1] + seg_b + route[j1:i2] + seg_a + route[j2:]
    new_routes[route_idx] = new_route

    return new_routes
