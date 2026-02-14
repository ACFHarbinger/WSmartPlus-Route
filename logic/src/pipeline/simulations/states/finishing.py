"""finishing.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import finishing
"""

from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING

import numpy as np

from logic.src.constants import DAY_METRICS, SIM_METRICS
from logic.src.utils.logging.log_utils import final_simulation_summary, log_to_json

from ..processor import save_matrix_to_excel
from .base import SimState

if TYPE_CHECKING:
    from .base import SimulationContext


class FinishingState(SimState):
    """State handles final result aggregation and persistence."""

    def handle(self, ctx: SimulationContext) -> None:
        """Handle.

        Args:
            ctx (SimulationContext): Description of ctx.
        """
        opts = ctx.opts
        assert ctx.bins is not None

        ctx.execution_time = time.process_time() - ctx.tic

        lg = [
            np.sum(ctx.bins.inoverflow),
            np.sum(ctx.bins.collected),
            np.sum(ctx.bins.ncollections),
            np.sum(ctx.bins.lost),
            ctx.bins.travel,
            (np.sum(ctx.bins.collected) / ctx.bins.travel if ctx.bins.travel > 0 else 0.0),
            np.sum(ctx.bins.inoverflow) - np.sum(ctx.bins.collected) + ctx.bins.travel,
            ctx.bins.profit,
            ctx.bins.ndays,
            ctx.execution_time,
        ]

        daily_log_path = os.path.join(
            ctx.results_dir,
            f"daily_{opts['data_distribution']}_{opts['n_samples']}N.json",
        )

        if opts["n_samples"] > 1:
            log_path = os.path.join(ctx.results_dir, f"log_full_{opts['n_samples']}N.json")
            log_to_json(
                log_path,
                SIM_METRICS,
                {ctx.policy: lg},
                sample_id=ctx.sample_id,
                lock=ctx.lock,
            )
            assert ctx.daily_log is not None
            log_to_json(
                daily_log_path,
                DAY_METRICS,
                {f"{ctx.pol_strip} #{ctx.sample_id}": ctx.daily_log.values()},
                lock=ctx.lock,
            )
        else:
            log_path = os.path.join(ctx.results_dir, f"log_mean_{opts['n_samples']}N.json")
            log_to_json(log_path, SIM_METRICS, {ctx.policy: lg}, lock=ctx.lock)
            assert ctx.daily_log is not None
            log_to_json(
                daily_log_path,
                DAY_METRICS,
                {ctx.pol_strip: ctx.daily_log.values()},
                lock=ctx.lock,
            )

        save_matrix_to_excel(
            ctx.bins.get_fill_history(),
            ctx.results_dir,
            opts["seed"],
            opts["data_distribution"],
            ctx.policy,
            ctx.sample_id,
        )

        if ctx.checkpoint:
            ctx.checkpoint.clear()

        # Clear shared metrics to prevent double counting in the progress bar
        if ctx.shared_metrics is not None:
            key = f"{ctx.policy}_{ctx.sample_id}"
            if key in ctx.shared_metrics:
                del ctx.shared_metrics[key]

        ctx.result = {ctx.policy: lg, "success": True}

        if opts.get("print_output"):
            final_simulation_summary({ctx.policy: lg}, ctx.policy, opts["n_samples"])

        ctx.transition_to(None)
