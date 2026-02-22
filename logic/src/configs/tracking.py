"""Tracking configuration dataclass for WSmart-Route."""

import os
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
        wandb_mode: Weights & Biases mode (``'online'``, ``'offline'``,
            ``'disabled'``).
        no_tensorboard: If ``True``, disable TensorBoard logging.
        no_progress_bar: If ``True``, disable the progress bar.
        log_dir: Directory to save logs.
        verbose: If ``True``, enable verbose logging.
        profile: If ``True``, enable function-level execution time profiling.
        log_step: Frequency of metric logging (e.g., every N steps).
        log_level: Logging level (e.g., ``'INFO'``, ``'DEBUG'``).
        real_time_log: If ``True``, enable real-time logging (e.g., dashboard).
        log_file: Path to the log file.
        instrument_ppo: Enabling detailed PPO diagnostics (clip_fraction, approx_kl, ratio, entropy).
        instrument_meta: Enabling Meta-RL tracking (meta rewards, feedbacks, optimizer loss).
        instrument_hpo: Enabling HPO trial-level logging (configs, metrics, runtime).
        instrument_rl_core: Enabling RL core diagnostics (policy_loss, log_likelihood, entropy).
        log_gradients: Enabling gradient norm logging.
        log_weights: Enabling model weight distribution logging.
    """

    wst_tracking_uri: str = "test_tracking" if os.environ.get("TEST_MODE") == "true" else "assets/tracking"
    mlflow_enabled: bool = False
    mlflow_tracking_uri: str = "test_mlruns" if os.environ.get("TEST_MODE") == "true" else "mlruns"
    mlflow_experiment_name: str = "wsmart-route"
    mlflow_run_name: Optional[str] = None
    ray_tune_storage_path: str = "ray_results"
    ray_tune_mlflow_enabled: bool = False
    zenml_enabled: bool = False
    zenml_store_url: str = ""
    zenml_stack_name: str = "wsmart-route-stack"

    # --- Logging & CLI Control (Migrated from root Config) ---
    wandb_mode: str = "offline"
    no_tensorboard: bool = False
    no_progress_bar: bool = False
    log_dir: str = "logs"
    verbose: bool = True
    profile: bool = False
    log_step: int = 10
    log_level: str = "INFO"
    real_time_log: bool = False
    log_file: Optional[str] = None

    # --- Instrumentation & Diagnostics ---
    instrument_ppo: bool = True
    instrument_meta: bool = True
    instrument_hpo: bool = True
    instrument_rl_core: bool = True
    log_gradients: bool = False
    log_weights: bool = False
