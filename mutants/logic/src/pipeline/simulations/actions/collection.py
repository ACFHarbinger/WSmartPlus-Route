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
        bins = context["bins"]
        tour = context["tour"]
        cost = context["cost"]

        collected, total_collected, ncol, profit = bins.collect(tour, cost)

        context["collected"] = collected
        context["total_collected"] = total_collected
        context["ncol"] = ncol
        context["profit"] = profit
