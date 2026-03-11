from typing import List

import numpy as np
from logic.src.policies.other.reinforcement_learning.local_search_manager import (
    LocalSearchManager,
)


class RVND:
    """
    Randomized Variable Neighborhood Descent.

    Explores multiple neighborhood structures in a completely random sequence.
    If a neighborhood yields an improvement, the search sequence is reset.
    Terminates when all neighborhoods have been explored without improvement.
    """

    def __init__(self, ls_manager: LocalSearchManager, rng: np.random.Generator):
        """
        Args:
            ls_manager: Global local search manager.
            rng: Random number generator for sequence exploration.
        """
        self.ls_manager = ls_manager
        self.rng = rng

        # We hook directly to the manager's built in core LS operators
        self.operators = [
            self.ls_manager.two_opt_intra,
            self.ls_manager.relocate,
            self.ls_manager.swap,
            self.ls_manager.swap_star,
            self.ls_manager.two_opt_star,
        ]

    def apply(self, routes: List[List[int]]) -> List[List[int]]:
        """
        Apply RVND until local optimum over all neighborhoods.

        Args:
            routes: Current list of node routes.

        Returns:
            List[List[int]]: Improved routes.
        """
        self.ls_manager.set_routes(routes)

        nl = list(range(len(self.operators)))
        while nl:
            # Pick a random local search neighborhood
            op_idx = self.rng.choice(nl)

            # Execute the shift/swap cascade
            improved = self.operators[op_idx]()

            if improved:
                # If we found a better route, reset the neighborhood list
                # because previously failing operators might work again on the new structure
                nl = list(range(len(self.operators)))
            else:
                # Discard operator if it hit local optima
                nl.remove(op_idx)

        return self.ls_manager.get_routes()
