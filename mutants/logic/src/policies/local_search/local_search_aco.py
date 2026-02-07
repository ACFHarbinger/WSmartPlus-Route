from typing import List

from .local_search_base import LocalSearch


class ACOLocalSearch(LocalSearch):
    """
    Local Search module for K-Sparse ACO.
    Implements 2-opt local search refinement.
    """

    def optimize(self, solution: List[List[int]]) -> List[List[int]]:
        """
        Apply full set of local search operators to the solution.

        Args:
            solution: List of routes, where each route is a list of node indices.

        Returns:
            List of optimized routes.
        """
        # Create a deep copy of routes to modify
        self.routes = [r[:] for r in solution]

        # Run optimization
        self._optimize_internal()

        return self.routes
