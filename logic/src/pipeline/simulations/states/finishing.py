"""finishing.py module.

Attributes:
    MODULE_VAR (Type): Description of module level variable.

Example:
    >>> import finishing
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from logic.src.constants import DAY_METRICS, SIM_METRICS
from logic.src.data.processor import save_matrix_to_excel
from logic.src.tracking.logging.log_utils import final_simulation_summary, log_to_json

try:
    from logic.src.tracking.core.run import get_active_run
    from logic.src.tracking.integrations.simulation import get_sim_tracker
except ImportError:
    get_active_run = None  # type: ignore[assignment,misc]
    get_sim_tracker = None  # type: ignore[assignment,misc]

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
            np.sum(ctx.bins.collected) - np.sum(ctx.bins.inoverflow) - ctx.bins.travel,
            ctx.bins.profit,
            ctx.execution_time,
            ctx.bins.ndays,
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

        # Log the fill-history export as a "save" dataset event
        with contextlib.suppress(Exception):
            run = get_active_run()
            if run is not None:
                excel_path = os.path.join(
                    ctx.results_dir,
                    "fill_history",
                    sim.data_distribution,
                    f"{ctx.pol_name}{sim.seed}_sample{ctx.sample_id}.xlsx",
                )
                run.log_dataset_event(
                    "save",
                    file_path=excel_path,
                    metadata={
                        "event": "fill_history_export",
                        "variable_name": "fill_history",
                        "policy": ctx.pol_name,
                        "sample_id": ctx.sample_id,
                        "source_file": "states/finishing.py",
                        "source_line": 90,
                    },
                )

        # Register all output files as tracking artifacts
        _log_result_artifacts(ctx, sim, log_path, daily_log_path)

        if ctx.checkpoint:
            ctx.checkpoint.clear()

        ctx.result = {ctx.pol_name: lg, "success": True}

        final_simulation_summary({ctx.pol_name: lg}, ctx.pol_name, sim.n_samples)

        # Forward final aggregated metrics to the centralised tracker (no-op if no run active)
        if get_sim_tracker is not None:
            sim_tracker = get_sim_tracker(ctx.pol_name, ctx.sample_id)
            if sim_tracker is not None:
                sim_tracker.log_final(SIM_METRICS, lg)

        ctx.transition_to(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _log_result_artifacts(ctx: Any, sim: Any, log_path: str, daily_log_path: str) -> None:
    """Register simulation output files with the active tracking run.

    Silently no-ops if no run is active or if any file doesn't exist yet.

    Registered artifact types:

    * ``result`` — primary JSON summary (per-sample full log or mean log)
    * ``result`` — per-day JSON log for this ``(policy, sample)`` pair
    * ``result`` — Excel fill-history matrix with metadata tag ``fill_history``
    """
    with contextlib.suppress(Exception):
        run = get_active_run()
        if run is None:
            return

        metadata = {"policy": ctx.pol_name, "sample_id": ctx.sample_id}

        for path in (log_path, daily_log_path):
            if os.path.exists(path):
                run.log_artifact(path, artifact_type="result", metadata=metadata)

        # Excel fill-history: {results_dir}/fill_history/{dist}/{policy}{seed}_sample{id}.xlsx
        excel_path = os.path.join(
            ctx.results_dir,
            "fill_history",
            sim.data_distribution,
            f"{ctx.pol_name}{sim.seed}_sample{ctx.sample_id}.xlsx",
        )
        if os.path.exists(excel_path):
            run.log_artifact(
                excel_path,
                artifact_type="result",
                metadata={**metadata, "type": "fill_history"},
            )
