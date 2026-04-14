"""
Steepest Or-opt Post-Processor.

Delegates to operators.intensification.or_opt_steepest (or its profit
variant when revenue/cost are configured) to perform chain relocations
(lengths 1, 2, 3) until a local minimum is reached.
"""

from typing import Any, List

from logic.src.interfaces.post_processing import IPostProcessor
from logic.src.policies.other.operators.intensification import (
    or_opt_steepest,
    or_opt_steepest_profit,
)

from .base import PostProcessorRegistry
from .common.helpers import assemble_tour, split_tour, to_numpy


@PostProcessorRegistry.register("or_opt_steepest")
class OrOptSteepestPostProcessor(IPostProcessor):
    """
    Steepest-descent Or-opt post-processor. Performs inter-route and
    intra-route chain relocations to improve the tour.
    """

    def process(self, tour: List[int], **kwargs: Any) -> List[int]:
        distance_matrix = kwargs.get("distance_matrix", kwargs.get("distancesC"))
        if distance_matrix is None or not tour:
            return tour

        wastes = kwargs.get("wastes", self.config.get("wastes", {}))
        capacity = kwargs.get("capacity", self.config.get("capacity", float("inf")))
        cost_per_km = kwargs.get("cost_per_km", self.config.get("cost_per_km", 0.0))
        revenue_kg = kwargs.get("revenue_kg", self.config.get("revenue_kg", 0.0))
        max_iter = kwargs.get("max_iter", self.config.get("max_iter", 500))
        chain_lengths_raw = kwargs.get("chain_lengths", self.config.get("chain_lengths", (1, 2, 3)))
        chain_lengths = tuple(chain_lengths_raw) if isinstance(chain_lengths_raw, (list, tuple)) else (1, 2, 3)

        dm = to_numpy(distance_matrix)

        try:
            routes = split_tour(tour)
            if not routes:
                return tour

            if revenue_kg > 0 or cost_per_km > 0:
                refined = or_opt_steepest_profit(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    R=revenue_kg,
                    C=cost_per_km,
                    max_iter=max_iter,
                    chain_lengths=chain_lengths,
                )
            else:
                refined = or_opt_steepest(
                    routes=routes,
                    dist_matrix=dm,
                    wastes=wastes,
                    capacity=capacity,
                    max_iter=max_iter,
                    chain_lengths=chain_lengths,
                )

            return assemble_tour(refined)

        except Exception:
            return tour
