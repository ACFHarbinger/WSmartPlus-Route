"""MLflow secondary sink for WSTracker runs.

This module provides the :class:`MLflowBridge` which attaches to a WSTracker run
and mirrors every metric, parameter, and artifact write to an MLflow tracking
server. It acts as a transparent secondary backend, ensuring fail-safe logging
even if the MLflow server is unreachable.

Attributes:
    MLflowBridge: A sink that forwards experiment data to MLflow.

Example:
    >>> bridge = MLflowBridge.attach(run, "http://localhost:5000", "VRPP-HNA")
    >>> run.log_metric("reward", 0.9)  # Mirrored to MLflow automatically
"""

from __future__ import annotations

import contextlib
from typing import Any, Dict, Optional

from logic.src.tracking.core.run import Run

try:
    import mlflow
except ImportError:
    mlflow = None  # type: ignore[assignment]


class MLflowBridge:
    """Forwards :class:`Run` events to an MLflow tracking server.

    Satisfies the sink protocol expected by the WSTracker run system. It
    encapsulates all MLflow communication, providing exception safety and
    asynchronous-like behavior by suppressing errors from the remote server.

    Attributes:
        _mlflow: Cached reference to the mlflow module.
        _active_run: The active MLflow run handle.
    """

    def __init__(
        self,
        mlflow_tracking_uri: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> None:
        """Initializes the MLflow bridge and starts a remote run.

        Args:
            mlflow_tracking_uri: URI of the MLflow tracking server.
            experiment_name: Name of the experiment in MLflow.
            run_name: Optional display name for the run.
            tags: Dictionary of tags to attach to the MLflow run.
        """
        self._mlflow = mlflow
        self._active_run: Optional[Any] = None
        if mlflow is not None:
            with contextlib.suppress(Exception):
                mlflow.set_tracking_uri(mlflow_tracking_uri)
                mlflow.set_experiment(experiment_name)
                self._active_run = mlflow.start_run(
                    run_name=run_name,
                    tags=tags or {},
                )

    # ------------------------------------------------------------------
    # Sink protocol
    # ------------------------------------------------------------------

    def log_metric(self, key: str, value: float, step: int) -> None:
        """Forwards a scalar metric to the MLflow tracking server.

        Args:
            key: Name of the metric.
            value: Scalar value.
            step: Training or evaluation step index.
        """
        if self._mlflow is None or self._active_run is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_metric(key, value, step=step)

    def log_params(self, params: Dict[str, Any]) -> None:
        """Forwards multiple parameters to the MLflow tracking server.

        Args:
            params: Mapping of parameter names to values (coerced to strings).
        """
        if self._mlflow is None or self._active_run is None:
            return
        with contextlib.suppress(Exception):
            # MLflow param values must be str; split into chunks of 100
            str_params = {k: str(v) for k, v in params.items()}
            items = list(str_params.items())
            for i in range(0, len(items), 100):
                self._mlflow.log_params(dict(items[i : i + 100]))

    def log_artifact(self, path: str) -> None:
        """Forwards a local file artifact to the MLflow storage backend.

        Args:
            path: Absolute path to the file to log.
        """
        if self._mlflow is None or self._active_run is None:
            return
        with contextlib.suppress(Exception):
            self._mlflow.log_artifact(path)

    def finish(self, status: str = "completed") -> None:
        """Terminates the MLflow run and updates the final status.

        Args:
            status: Final status of the WSTracker run.
        """
        if self._mlflow is None or self._active_run is None:
            return
        with contextlib.suppress(Exception):
            mlflow_status = "FINISHED" if status == "completed" else "FAILED"
            self._mlflow.end_run(status=mlflow_status)
            self._active_run = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def attach(
        cls,
        run: Run,
        mlflow_tracking_uri: str,
        experiment_name: str,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> MLflowBridge:
        """Factory method to create and attach a bridge to an existing run.

        Args:
            run: The WSTracker run instance to mirror.
            mlflow_tracking_uri: URI of the MLflow tracking server.
            experiment_name: MLflow experiment name.
            run_name: Optional display name for the run.
            tags: Optional tags to attach to the MLflow run.

        Returns:
            MLflowBridge: The initialized and attached bridge instance.
        """
        bridge = cls(
            mlflow_tracking_uri=mlflow_tracking_uri,
            experiment_name=experiment_name,
            run_name=run_name or run.run_id[:8],
            tags=tags,
        )
        run.add_sink(bridge)
        return bridge
