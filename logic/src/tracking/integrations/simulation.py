"""Simulation pipeline hooks for WSTracker.

This module provides :class:`SimulationRunTracker`, a lightweight helper that
wraps an active WSTracker run and adds simulation-specific logging conventions.
It ensures that multi-policy and multi-sample simulation experiments are
correctly namespaced and traceable in the experiment database.

Attributes:
    SimulationRunTracker: Helper for namespaced logging in simulation runs.

Example:
    >>> tracker = get_sim_tracker("ALNS", 42)
    >>> if tracker:
    ...     tracker.log_day(1, {"profit": 1500.0})
"""

from __future__ import annotations

import contextlib
import time
from typing import Any, Dict, List, Optional

from logic.src.tracking.core.run import Run, get_active_run


class SimulationRunTracker:
    """Simulation-specific metric logger wrapping an active :class:`Run`.

    Provides a namespaced interface for logging per-day metrics and final
    aggregates. All keys are prefixed with ``{policy}/s{sample}`` to allow
    multiple independent simulation evaluations to be recorded within the
    same parent experiment run.

    Attributes:
        _run: The active tracking run being used as a sink.
        policy_name: Name of the evaluation policy.
        sample_id: Index of the current simulation seed/sample.
        _start_wall: Monotonic start time for calculation of wall-clock duration.
        _prefix: Precomputed namespace prefix for metric keys.
    """

    def __init__(self, run: Run, policy_name: str, sample_id: int) -> None:
        """Initializes the simulation tracker.

        Args:
            run: The active tracking run.
            policy_name: Name of the routing policy.
            sample_id: Numeric index of the simulation sample.
        """
        self._run = run
        self.policy_name = policy_name
        self.sample_id = sample_id
        self._start_wall = time.monotonic()
        self._prefix = f"{policy_name}/s{sample_id}"

    # ------------------------------------------------------------------
    # Per-day logging
    # ------------------------------------------------------------------

    def log_day(self, day: int, metrics: Dict[str, Any]) -> None:
        """Logs scalar metrics for a single simulation day.

        Args:
            day: Current simulation day (used as step).
            metrics: Dictionary of metrics to log. Non-numeric values are skipped.
        """
        tagged: Dict[str, float] = {}
        for k, v in metrics.items():
            with contextlib.suppress(TypeError, ValueError):
                tagged[f"{self._prefix}/{k}"] = float(v)
        if tagged:
            self._run.log_metrics(tagged, step=day)

    # ------------------------------------------------------------------
    # Final aggregated result
    # ------------------------------------------------------------------

    def log_final(self, metric_keys: List[str], metric_values: List[Any]) -> None:
        """Logs end-of-simulation summary metrics and total wall-clock time.

        Args:
            metric_keys: Names of the aggregated metrics.
            metric_values: Scalar values corresponding to the keys.
        """
        prefix = f"{self._prefix}/final"
        for key, val in zip(metric_keys, metric_values, strict=False):
            with contextlib.suppress(TypeError, ValueError):
                self._run.log_metric(f"{prefix}/{key}", float(val), step=self.sample_id)

        elapsed = time.monotonic() - self._start_wall
        self._run.log_metric(f"{prefix}/wall_time_s", elapsed, step=self.sample_id)
        self._run.flush()

    # ------------------------------------------------------------------
    # Error recording
    # ------------------------------------------------------------------

    def log_failure(self, error: str) -> None:
        """Logs a failure tag to the run for documentation of crashed samples.

        Args:
            error: Descriptive error message or stack trace snippet.
        """
        tag_key = f"error.{self.policy_name}.s{self.sample_id}"
        self._run.set_tag(tag_key, error[:250])


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def get_sim_tracker(
    policy_name: str,
    sample_id: int,
) -> Optional[SimulationRunTracker]:
    """Obtains a properly initialized simulation tracker for the active run.

    This factory is the preferred method for workers to initialize tracking
    without requiring direct access to a global tracker instance.

    Args:
        policy_name: Name of the policy.
        sample_id: Unique sample index.

    Returns:
        Optional[SimulationRunTracker]: An initialized tracker, or None if no
            active run is detected.
    """
    run = get_active_run()
    if run is None:
        return None
    return SimulationRunTracker(run, policy_name, sample_id)
