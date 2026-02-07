from __future__ import annotations

from typing import TYPE_CHECKING, Dict

import numpy as np

from .local_search_base import LocalSearch

if TYPE_CHECKING:
    from ..hybrid_genetic_search import HGSParams, Individual


class HGSLocalSearch(LocalSearch):
    """
    Local Search module for HGS.
    """

    def __init__(
        self,
        dist_matrix: np.ndarray,
        demands: Dict[int, float],
        capacity: float,
        R: float,
        C: float,
        params: "HGSParams",
    ):
        super().__init__(dist_matrix, demands, capacity, R, C, params)

    def optimize(self, solution: "Individual") -> "Individual":
        """
        Iteratively improve an individual using local search operators.
        """
        if not solution.routes:
            return solution

        self.routes = [r[:] for r in solution.routes]

        # Run optimization
        self._optimize_internal()

        solution.routes = self.routes
        gt = []
        for r in self.routes:
            gt.extend(r)
        solution.giant_tour = gt
        return solution
