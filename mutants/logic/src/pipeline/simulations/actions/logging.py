"""
Action for simulation logging.
"""

from typing import Any, Dict

from logic.src.utils.logging.log_utils import send_daily_output_to_gui

from ..day_context import get_daily_results
from .base import SimulationAction


class LogAction(SimulationAction):
    """
    Records daily simulation metrics and outputs results.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Log daily results and update GUI."""
        tour = context["tour"]
        cost = context["cost"]
        total_collected = context["total_collected"]
        ncol = context["ncol"]
        profit = context["profit"]
        new_overflows = context["new_overflows"]
        coords = context["coords"]
        day = context["day"]
        sum_lost = context["sum_lost"]

        dlog = get_daily_results(
            total_collected,
            ncol,
            cost,
            tour,
            day,
            new_overflows,
            sum_lost,
            coords,
            profit,
        )

        context["daily_log"] = dlog

        send_daily_output_to_gui(
            dlog,
            context["policy_name"],
            context["sample_id"],
            context["day"],
            context["total_fill"],
            context["collected"],
            context["bins"].real_c,
            context["realtime_log_path"],
            tour,
            coords,
            context["lock"],
            must_go=context.get("must_go"),
        )
