"""
Action for daily bin filling.
"""

from typing import Any, Dict

from .base import SimulationAction


class FillAction(SimulationAction):
    """
    Executes daily bin filling simulation.

    Queries the bin management object to update fill levels based on
    either stochastic generation or empirical data from files.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute daily bin filling."""
        bins = context["bins"]
        day = context["day"]

        if bins.is_stochastic():
            new_overflows, fill, total_fill, sum_lost = bins.stochasticFilling()
        else:
            new_overflows, fill, total_fill, sum_lost = bins.loadFilling(day)
        context["new_overflows"] = new_overflows
        context["fill"] = fill
        context["total_fill"] = total_fill
        context["sum_lost"] = sum_lost

        # Accumulate overflows in context (if needed by subsequent steps or final return)
        if "overflows" in context:
            context["overflows"] += new_overflows
