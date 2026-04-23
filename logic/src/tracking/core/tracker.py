"""Central experiment tracker — the main entry-point of WSTracker.

This module provides the central coordination for tracking experiments. It
manages the instantiation of the tracking store and orchestrates the
creation and retrieval of tracked runs.

Attributes:
    Tracker: Central coordinator for experiment tracking.
    get_tracker: Utility for singleton-style access to the active tracker.

Example:
    >>> from logic.src.tracking.core.tracker import Tracker
    >>> tracker = Tracker(tracking_uri="./mlruns")
    >>> with tracker.start_run("Optimization-v1") as run:
    ...     run.log_param("optimizer", "adam")
"""

from __future__ import annotations

import os
import uuid
from typing import Any, Dict, List, Optional

from .run import Run, set_active_run
from .store import TrackingStore

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_tracker: Optional["Tracker"] = None


def get_tracker() -> Optional["Tracker"]:
    """Returns the process-global Tracker instance.

    Usually initialized by the training entry point via the wst.init() factory.

    Returns:
        Optional[Tracker]: The current process-global tracker instance, or
            None if not yet initialized.
    """
    return _tracker


# ---------------------------------------------------------------------------
# Tracker
# ---------------------------------------------------------------------------


class Tracker:
    """Centralized experiment tracker for WSmart-Route.

    A single Tracker instance manages the SQLite database connectivity and
    coordinates the creation of Run objects. It acts as the primary
    administration layer for experiments and runs, ensuring that artifacts
    and metrics are logically grouped.

    Attributes:
        tracking_uri: Directory holding 'tracking.db' and the 'artifacts/' tree.
        buffer_size: Default metric buffer size for runs created by this tracker.
    """

    def __init__(self, tracking_uri: str, buffer_size: int = 200) -> None:
        """Initializes a new experiment tracker.

        Args:
            tracking_uri: Directory that will hold the tracking database and
                the artifact storage sub-tree.
            buffer_size: Number of metric points to buffer per run before
                flushing to SQLite. Defaults to 200.
        """
        self.tracking_uri = tracking_uri
        self.buffer_size = buffer_size

        # Ensure base directory exists so SQLite can create the tracking.db file
        os.makedirs(tracking_uri, exist_ok=True)

        db_path = os.path.join(tracking_uri, "tracking.db")
        self._store = TrackingStore(db_path)

        self._artifacts_root = os.path.join(tracking_uri, "artifacts")
        os.makedirs(self._artifacts_root, exist_ok=True)

    # ------------------------------------------------------------------
    # Run creation
    # ------------------------------------------------------------------

    def start_run(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        run_type: str = "generic",
        tags: Optional[Dict[str, str]] = None,
        description: str = "",
    ) -> Run:
        """Creates a new run record and returns an active run instance.

        The returned run object is intended to be used as a context manager to
        ensure status is correctly updated on completion or failure.

        Args:
            experiment_name: Name of the parent experiment. Created if it
                does not exist.
            run_name: Optional human-readable label for the run.
            run_type: Category identifier (e.g., 'training', 'simulation',
                'evaluation', 'data_gen').
            tags: Initial metadata tags to attach to the run.
            description: Detailed text for the experiment. Only applied if
                creating a new experiment.

        Returns:
            Run: A fresh tracking run instance.
        """
        exp_id = self._store.get_or_create_experiment(experiment_name, description)
        run_id = str(uuid.uuid4())

        artifact_dir = os.path.join(self._artifacts_root, experiment_name, run_id)
        os.makedirs(artifact_dir, exist_ok=True)

        self._store.create_run(run_id, exp_id, run_name, run_type, artifact_dir)

        if tags:
            self._store.set_tags(run_id, {k: str(v) for k, v in tags.items()})

        run = Run(run_id, self._store, artifact_dir, self.buffer_size)
        return run

    def attach_run(self, run_id: str) -> Run:
        """Attaches to an existing run ID and registers it as active.

        Specifically useful for distributed workers or sub-processes that need
        to log back into a run initiated by a parent process. This restores
        the artifact directory context from the database.

        Args:
            run_id: Unique UUID string identifying the target run.

        Returns:
            Run: A tracking run instance re-attached to the existing database record.

        Raises:
            ValueError: If the provided run_id cannot be found in the store.
        """
        run_data = self._store.get_run(run_id)
        if run_data is None:
            raise ValueError(f"Run '{run_id}' not found in tracking store at {self.tracking_uri!r}")
        artifact_dir = run_data.get("artifact_dir") or ""
        run = Run(run_id, self._store, artifact_dir, self.buffer_size)
        set_active_run(run)
        return run

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def list_experiments(self) -> List[Dict[str, Any]]:
        """Retrieves a list of all experiments managed by this tracker.

        Returns:
            List[Dict[str, Any]]: Experiment records ordered by creation date.
        """
        return self._store.list_experiments()

    def list_runs(
        self,
        experiment_name: Optional[str] = None,
        run_type: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieves a filtered list of runs across all experiments.

        Args:
            experiment_name: Optional filter to restrict results to a specific
                experiment name.
            run_type: Optional filter by run category (e.g., 'training').
            status: Optional filter by terminal status (e.g., 'completed').

        Returns:
            List[Dict[str, Any]]: Matching run records ordered by start time.
        """
        exp_id: Optional[int] = None
        if experiment_name:
            for exp in self._store.list_experiments():
                if exp["name"] == experiment_name:
                    exp_id = int(exp["id"])
                    break
        return self._store.list_runs(experiment_id=exp_id, run_type=run_type, status=status)

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves the record for a specific run.

        Args:
            run_id: Unique UUID string identifying the run.

        Returns:
            Optional[Dict[str, Any]]: The run record dictionary containing
                metadata and artifact locations, or None if not found.
        """
        return self._store.get_run(run_id)
