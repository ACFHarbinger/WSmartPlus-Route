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
from logic.src.tracking.integrations.simulation import get_sim_tracker
from logic.src.tracking.logging.log_utils import final_simulation_summary, log_to_json

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
        sim = ctx.cfg.sim
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
            f"daily_{sim.data_distribution}_{sim.n_samples}N.json",
        )

        if sim.n_samples > 1:
            log_path = os.path.join(ctx.results_dir, f"log_full_{sim.n_samples}N.json")
            log_to_json(
                log_path,
                SIM_METRICS,
                {ctx.pol_name: lg},
                sample_id=ctx.sample_id,
                lock=ctx.lock,
            )
            assert ctx.daily_log is not None
            log_to_json(
                daily_log_path,
                DAY_METRICS,
                {f"{ctx.pol_name} #{ctx.sample_id}": ctx.daily_log.values()},
                lock=ctx.lock,
            )
        else:
            log_path = os.path.join(ctx.results_dir, f"log_mean_{sim.n_samples}N.json")
            log_to_json(log_path, SIM_METRICS, {ctx.pol_name: lg}, lock=ctx.lock)
            assert ctx.daily_log is not None
            log_to_json(
                daily_log_path,
                DAY_METRICS,
                {ctx.pol_name: ctx.daily_log.values()},
                lock=ctx.lock,
            )

        save_matrix_to_excel(
            ctx.bins.get_fill_history(),
            ctx.results_dir,
            sim.seed,
            sim.data_distribution,
            ctx.pol_name,
            ctx.sample_id,
        )

        if ctx.checkpoint:
            ctx.checkpoint.clear()

        # Clear shared metrics to prevent double counting in the progress bar
        if ctx.shared_metrics is not None:
            key = f"{ctx.pol_name}_{ctx.sample_id}"
            if key in ctx.shared_metrics:
                del ctx.shared_metrics[key]

        ctx.result = {ctx.pol_name: lg, "success": True}

        final_simulation_summary({ctx.pol_name: lg}, ctx.pol_name, sim.n_samples)

        # Forward final aggregated metrics to the centralised tracker (no-op if no run active)
        sim_tracker = get_sim_tracker(ctx.pol_name, ctx.sample_id)
        if sim_tracker is not None:
            sim_tracker.log_final(SIM_METRICS, lg)

        ctx.transition_to(None)
