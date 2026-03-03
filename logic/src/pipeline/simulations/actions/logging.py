"""
Action for simulation logging.
"""

from typing import Any, Dict

from logic.src.tracking.integrations.simulation import get_sim_tracker
from logic.src.tracking.logging.log_utils import send_daily_output_to_gui

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
        time = context["time"]

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
            time,
        )

        context["daily_log"] = dlog
        policy_name = context["policy_name"]
        print(f"[INFO] Logging daily results for {policy_name} policy.")
        send_daily_output_to_gui(
            dlog,
            policy_name,
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

        # Forward day-level metrics to the centralised tracker (no-op if no run active)
        sim_tracker = get_sim_tracker(context["policy_name"], context["sample_id"])
        if sim_tracker is not None:
            sim_tracker.log_day(
                day,
                {
                    "cost": cost,
                    "profit": profit,
                    "ncol": ncol,
                    "kg": total_collected,
                    "overflows": new_overflows,
                    "kg_lost": sum_lost,
                    "time": time,
                },
            )
