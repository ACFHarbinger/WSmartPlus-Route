from typing import List

import numpy as np

from logic.src.policies.other.local_search.local_search_aco import (
    ACOLocalSearch,
)


class RVND:
    """
    Randomized Variable Neighborhood Descent.

    Explores multiple neighborhood structures in a completely random sequence.
    If a neighborhood yields an improvement, the search sequence is reset.
    Terminates when all neighborhoods have been explored without improvement.
    """

    def __init__(self, ls_manager: ACOLocalSearch, rng: np.random.Generator):
        """
        Args:
            ls_manager: Global local search manager.
            rng: Random number generator (kept for API consistency, though ACOLocalSearch handles randomization).
        """
        self.ls_manager = ls_manager
        self.rng = rng

    def apply(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply RVND until local optimum over all neighborhoods.

        Args:
            routes: Current list of node routes.

        Returns:
            List[List[int]]: Improved routes.
        """
        return self.ls_manager.optimize(routes)
