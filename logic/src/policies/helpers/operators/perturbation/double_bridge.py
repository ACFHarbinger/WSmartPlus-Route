"""
Double-Bridge Move (Random 4-opt) Perturbation Module.

Implements the double-bridge move, a classical 4-opt perturbation that
slices a route into 4 segments by choosing 3 random cut points and
reconnects them in a non-sequential order to escape local optima.

Given segments A, B, C, D the reconnection is: A + C + B + D.
This creates a configuration that cannot be reached by any sequence of
2-opt or 3-opt moves, making it a powerful diversification tool.

Attributes:
    None

Example:
    >>> from logic.src.policies.helpers.operators.perturbation.double_bridge import double_bridge
    >>> improved = double_bridge(ls, r_idx=0, rng=rng)
"""

from random import Random
from typing import Any, Optional


def double_bridge(ls: Any, r_idx: int, rng: Optional[Random] = None) -> bool:
    """
    Double-bridge (4-opt) perturbation on a single route.

    Selects 3 random cut points, slicing the route into 4 segments
    A, B, C, D and reconnects as A + C + B + D.

    This is always applied (no improving check) since it is a
    perturbation operator designed to escape local optima.

    Args:
        ls: LocalSearch instance.
        r_idx: Route index.
        rng: Random number generator.

    Returns:
        bool: True if the move was applied (route long enough), False otherwise.
    """
    if rng is None:
        rng = Random()

    route = ls.routes[r_idx]
    n = len(route)

    if n < 4:
        return False

    # Generate 3 sorted cut points in [1, n-1)
    cuts = sorted(rng.sample(range(1, n), min(3, n - 1)))
    if len(cuts) < 3:
        return False

    c1, c2, c3 = cuts[0], cuts[1], cuts[2]

    seg_a = route[:c1]
    seg_b = route[c1:c2]
    seg_c = route[c2:c3]
    seg_d = route[c3:]

    # Reconnect: A + C + B + D
    ls.routes[r_idx] = seg_a + seg_c + seg_b + seg_d
    ls._update_map({r_idx})
    return True
