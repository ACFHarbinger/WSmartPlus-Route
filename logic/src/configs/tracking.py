"""Tracking configuration dataclass for WSmart-Route."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class TrackingConfig:
    """Centralised tracking backend configuration.

    Controls both the native WSTracker (SQLite) and the optional MLflow
    secondary sink.  When ``mlflow_enabled`` is ``True``, every metric,
    parameter, and artifact logged to WSTracker is also mirrored to the
    configured MLflow server.

    Attributes:
        wst_tracking_uri: Directory that holds ``tracking.db`` and the
            ``artifacts/`` subtree for the native WSTracker backend.
        mlflow_enabled: When ``True``, attach an :class:`MLflowBridge`
            sink to every new run so data is dual-written to MLflow.
        mlflow_tracking_uri: MLflow tracking server URI.  Supports local
            paths (``mlruns``), ``http://`` addresses, and Databricks
            workspace URIs.
        mlflow_experiment_name: MLflow experiment name to create/reuse.
        mlflow_run_name: Optional human-readable name shown in the MLflow
            UI.  Defaults to the first 8 characters of the WSTracker
            run UUID when ``None``.
        ray_tune_storage_path: Root directory for Ray Tune trial logs and
            checkpoints.  Defaults to ``ray_results/`` in the project root.
        ray_tune_mlflow_enabled: When ``True``, each Ray Tune trial also
            logs to MLflow via ``MLflowLoggerCallback``.  Requires
            ``mlflow_enabled`` to also be ``True`` for the URI/experiment
            settings to be available.
        zenml_enabled: When ``True``, training, simulation, and HPO runs are
            orchestrated through versioned ZenML pipelines instead of being
            dispatched directly.  Requires ``zenml`` to be installed
            (``uv add "zenml[mlflow]"``).
        zenml_store_url: ZenML metadata store URL.  An empty string uses the
            default local store (``~/.zenml``).  Supported formats:
            ``sqlite:///path/to/store.db`` and
            ``mysql://user:password@host/database``.
        zenml_stack_name: Name of the ZenML stack to activate for pipeline
            runs.  The stack must include an ``mlflow_tracker`` experiment
            tracker component pointing at ``mlflow_tracking_uri``.
    """

    wst_tracking_uri: str = "assets/tracking"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "mlruns"
    mlflow_experiment_name: str = "wsmart-route"
    mlflow_run_name: Optional[str] = None
    ray_tune_storage_path: str = "ray_results"
    ray_tune_mlflow_enabled: bool = False
    zenml_enabled: bool = False
    zenml_store_url: str = ""
    zenml_stack_name: str = "wsmart-route-stack"
