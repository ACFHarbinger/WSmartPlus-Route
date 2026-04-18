"""
Action for simulation logging.
"""

from typing import Any, Dict

from logic.src.pipeline.simulations.actions.base import SimulationAction
from logic.src.pipeline.simulations.day_context import get_daily_results
from logic.src.tracking.integrations.simulation import get_sim_tracker
from logic.src.tracking.logging.log_utils import send_daily_output_to_gui


class LogAction(SimulationAction):
    """
    Records daily simulation metrics and outputs results.
    """

    def execute(self, context: Dict[str, Any]) -> None:
        """Log daily results and update GUI."""
        tour = context["tour"]
        km = context["cost"]
        total_collected = context["total_collected"]
        ncol = context["ncol"]
        profit = context["profit"]
        new_overflows = context["new_overflows"]
        coords = context["coords"]
        day = context["day"]
        sum_lost = context["sum_lost"]
        time = context["time"]
        day = context["day"]

        dlog = get_daily_results(
            total_collected,
            ncol,
            km,
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
        print(f"\n[INFO] Logging day {day} results for {policy_name} policy.")
        send_daily_output_to_gui(
            dlog,
            policy_name,
            context["sample_id"],
            day,
            context["total_fill"],
            context["collected"],
            context["bins"].real_c,
            context["realtime_log_path"],
            tour,
            coords,
            context["lock"],
            mandatory=context.get("mandatory"),
        )

        # Forward day-level metrics to the centralised tracker (no-op if no run active)
        sim_tracker = get_sim_tracker(context["policy_name"], context["sample_id"])
        if sim_tracker is not None:
            sim_tracker.log_day(
                day,
                {
                    "reward": dlog["reward"],
                    "profit": profit,
                    "ncol": ncol,
                    "kg": total_collected,
                    "overflows": new_overflows,
                    "km": km,
                    "kg_lost": sum_lost,
                    "time": time,
                },
            )
