"""
Action for waste collection execution.
"""

from typing import Any, Dict

from .base import SimulationAction


class CollectAction(SimulationAction):
    """
    Processes waste collection from bins visited in the tour.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute waste collection based on the generated tour."""
        from logic.src.policies.route_construction.other_algorithms.travelling_salesman_problem.tsp import (
            get_route_cost,
        )

        bins = context["bins"]
        tour = context["tour"]

        # 1. METRIC CONSISTENCY: Always re-calculate KM from the final tour
        # This combines mandatory selection, construction, and any route improvements.
        # We use the raw distance matrix from the context (guaranteed to be KM).
        dist_matrix = context["distance_matrix"]
        raw_km = get_route_cost(dist_matrix, tour)

        # 2. Perform collection using strictly KM-based cost
        # Bins.collect internally handles normalized revenue (collected mass * €/kg)
        # and expenses (raw_km * €/km).
        collected, total_collected, ncol, profit = bins.collect(tour, raw_km)

        # 3. Update context with definitive source-of-truth metrics for LogAction
        context["cost"] = raw_km
        context["collected"] = collected
        context["total_collected"] = total_collected
        context["ncol"] = ncol
        context["profit"] = profit
