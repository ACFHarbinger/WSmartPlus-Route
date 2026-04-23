"""Data-lineage tracking callback for simulation runs.

This module provides the :class:`DataLineageCallback` which hooks into the
simulation loop to record per-day KPI scalars and periodic snapshots of
bin fill level distributions. It ensures full traceability of data evolution
throughout multi-day simulation episodes.

Attributes:
    DataLineageCallback: Step-level data lineage and KPI tracker for simulations.

Example:
    >>> cb = DataLineageCallback(policy_name="HNA", sample_id=0, log_freq=5)
    >>> cb.on_simulation_start(sim_context)
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import torch

from logic.src.tracking.core.run import get_active_run
from logic.src.tracking.integrations.data import RuntimeDataTracker
from logic.src.tracking.integrations.simulation import SimulationRunTracker

if TYPE_CHECKING:
    from logic.src.tracking.integrations.simulation import SimulationRunTracker


class DataLineageCallback:
    """Step-level data lineage and KPI tracker for a simulation run.

    Records two complementary streams per policy/sample pair: per-day KPIs
    (profit, distance, etc.) and throttled bin fill distribution snapshots
    for drift analysis.

    Attributes:
        _policy_name: Name of the policy being evaluated.
        _sample_id: Unique index for the current simulation sample.
        _log_freq: Frequency (in days) for distribution snapshots.
        _step_count: Internal counter of simulation steps.
        _data_tracker: Instance of RuntimeDataTracker for distribution logging.
        _sim_tracker: Instance of SimulationRunTracker for KPI logging.
    """

    def __init__(self, policy_name: str, sample_id: int, log_freq: int = 1) -> None:
        """Initializes the data lineage callback.

        Args:
            policy_name: Name of the policy for namespacing metrics.
            sample_id: Sample index for namespacing metrics.
            log_freq: Frequency of distribution snapshots in simulation days.
                Defaults to 1.
        """
        self._policy_name = policy_name
        self._sample_id = sample_id
        self._log_freq = max(1, log_freq)
        self._step_count: int = 0
        self._data_tracker: Optional[RuntimeDataTracker] = None
        self._sim_tracker: Optional[SimulationRunTracker] = None

    # ------------------------------------------------------------------
    # Public lifecycle hooks
    # ------------------------------------------------------------------

    def on_simulation_start(self, context: Any) -> None:
        """Initializes trackers and snapshots the baseline bin distribution.

        Args:
            context: The simulation context containing initialized bins.
        """
        run = get_active_run()
        if run is None:
            return

        self._data_tracker = RuntimeDataTracker(run)
        self._sim_tracker = SimulationRunTracker(run, self._policy_name, self._sample_id)

        bins = getattr(context, "bins", None)
        if bins is not None:
            fill_dict = _bins_to_tensor_dict(bins)
            if fill_dict:
                self._data_tracker.on_load(
                    fill_dict,
                    shape=next(iter(fill_dict.values())).shape,
                    metadata={
                        "event": "simulation_start",
                        "policy": self._policy_name,
                        "sample_id": self._sample_id,
                        "variable_name": "bins_fill_distribution",
                        "source_file": "integrations/data_lineage.py",
                        "source_line": 95,
                    },
                )

    def on_step_end(self, day_context: Any, day: int) -> None:
        """Logs daily KPIs and throttled distribution snapshots.

        Args:
            day_context: Context for the current simulation day.
            day: Current simulation day index.
        """
        self._step_count += 1

        # --- 1. Log scalar KPIs every day --------------------------------
        if self._sim_tracker is not None:
            dlog: Dict[str, Any] = getattr(day_context, "daily_log", None) or {}
            self._sim_tracker.log_day(day, dlog)

            # Tour length (list → count of stops, not loggable as float directly)
            tour = dlog.get("tour")
            if tour is not None:
                with contextlib.suppress(Exception):
                    self._sim_tracker._run.log_metric(
                        f"{self._sim_tracker._prefix}/tour_len", float(len(tour)), step=day
                    )

            # Cumulative bins scalars not present in dlog (km, profit, kg totals).
            bins = getattr(day_context, "bins", None)
            if bins is not None:
                _log_cumulative_bins(self._sim_tracker, bins, day)

        # --- 2. Throttled fill-distribution snapshot ---------------------
        if self._step_count % self._log_freq != 0:
            return

        if self._data_tracker is not None:
            bins = getattr(day_context, "bins", None)
            if bins is not None:
                fill_dict = _bins_to_tensor_dict(bins)
                if fill_dict:
                    self._data_tracker.snapshot(fill_dict, tag=f"sim/day_{day}", step=day)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log_cumulative_bins(sim_tracker: SimulationRunTracker, bins: Any, day: int) -> None:
    """Logs cumulative bin state scalars (km, kg, profit) as time-series metrics.

    Args:
        sim_tracker: The active simulation KPI tracker.
        bins: The bins object containing cumulative state.
        day: Current simulation day.
    """
    prefix = f"{sim_tracker._prefix}/cumul"
    scalars: Dict[str, float] = {}

    with contextlib.suppress(Exception):
        travel = getattr(bins, "travel", None)
        if travel is not None:
            scalars[f"{prefix}/km"] = float(travel)

    with contextlib.suppress(Exception):
        collected = getattr(bins, "collected", None)
        if collected is not None:
            scalars[f"{prefix}/kg"] = float(np.sum(collected))

        lost = getattr(bins, "lost", None)
        if lost is not None:
            scalars[f"{prefix}/kg_lost"] = float(np.sum(lost))

        inoverflow = getattr(bins, "inoverflow", None)
        if inoverflow is not None:
            scalars[f"{prefix}/overflows"] = float(np.sum(inoverflow))

    with contextlib.suppress(Exception):
        profit = getattr(bins, "profit", None)
        if profit is not None:
            scalars[f"{prefix}/profit"] = float(profit)

    if scalars:
        sim_tracker._run.log_metrics(scalars, step=day)


def _bins_to_tensor_dict(bins: Any) -> Dict[str, torch.Tensor]:
    """Converts bin numpy arrays to a tensor dictionary for tracking.

    Args:
        bins: The bins object to extract data from.

    Returns:
        Dict[str, torch.Tensor]: Mapping of field names to torch tensors.
    """
    mapping = {
        "fill": "c",
        "real_fill": "real_c",
        "lost": "lost",
        "inoverflow": "inoverflow",
    }
    result: Dict[str, torch.Tensor] = {}
    for tensor_name, attr in mapping.items():
        with contextlib.suppress(Exception):
            arr = getattr(bins, attr, None)
            if arr is not None:
                result[tensor_name] = torch.as_tensor(arr, dtype=torch.float32)
    return result
