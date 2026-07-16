"""Read service for backend-agnostic experiment tracking.

This module provides cached Streamlit functions to query metrics and runs
from multiple tracking backends including the native WSTracker (SQLite),
MLflow, and ZenML. It enables unified visualization of simulation stability
metrics and deep learning training progress.

Attributes:
    load_tracking_runs: Fetches runs from the native SQLite store.
    load_mlflow_runs: Fetches runs from an MLflow tracking server.
    load_zenml_pipeline_runs: Fetches recent ZenML pipeline executions.

Example:
    >>> from logic.src.ui.services.tracking_service import load_tracking_runs
    >>> runs = load_tracking_runs(tracking_uri="assets/tracking")
    >>> print(f"Found {len(runs)} active experiments.")
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import streamlit as st

try:
    import mlflow
    from mlflow.entities import Run
    from mlflow.tracking import MlflowClient
except ImportError:
    mlflow = None  # type: ignore[assignment]
    Run = Any  # type: ignore[misc, assignment]
    MlflowClient = Any  # type: ignore[misc, assignment]

try:
    from zenml.client import Client as ZenMLClient
except ImportError:
    ZenMLClient = Any  # type: ignore[misc, assignment]

with contextlib.suppress(ImportError):
    from logic.src.tracking.core.store import TrackingStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tracking_uri(tracking_uri: Optional[str]) -> str:
    """Resolves the tracking URI, falling back to the default assets path.

    Args:
        tracking_uri: Optional user-provided path or URI.

    Returns:
        str: Resolved path to the tracking directory.
    """
    if tracking_uri:
        return tracking_uri
    return str(Path.cwd() / "assets" / "tracking")


def _open_store(tracking_uri: Optional[str]) -> Optional[Any]:
    """Retrieves a TrackingStore instance for the specified URI.

    Args:
        tracking_uri: Path to the tracking directory.

    Returns:
        Optional[TrackingStore]: An initialized store if valid, else None.
    """
    uri = _get_tracking_uri(tracking_uri)
    db_path = os.path.join(uri, "tracking.db")
    if not os.path.exists(db_path):
        return None
    with contextlib.suppress(Exception):
        return TrackingStore(db_path)
    return None


# ---------------------------------------------------------------------------
# Cached queries
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def load_tracking_runs(
    tracking_uri: Optional[str] = None,
    run_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves all experiment runs from the native WSTracker database.

    Args:
        tracking_uri: Path containing tracking.db. Defaults to assets/tracking.
        run_type: Optional filter (e.g., "training" or "simulation").

    Returns:
        List[Dict[str, Any]]: Sequence of run metadata dictionaries.
    """
    store = _open_store(tracking_uri)
    if store is None:
        return []
    with contextlib.suppress(Exception):
        return store.list_runs(run_type=run_type)
    return []


@st.cache_data(ttl=30)
def load_run_metrics(
    run_id: str,
    metric_key: str,
    tracking_uri: Optional[str] = None,
) -> pd.DataFrame:
    """Retrieves the step-indexed history of a specific metric for a run.

    Args:
        run_id: Unique identifier for the tracking run.
        metric_key: identifier for the metric (e.g., "gurobi/s0/profit").
        tracking_uri: Path to the tracking directory.

    Returns:
        pd.DataFrame: History with columns [step, value, timestamp].
    """
    store = _open_store(tracking_uri)
    if store is None:
        return pd.DataFrame(columns=["step", "value", "timestamp"])
    with contextlib.suppress(Exception):
        rows = store.get_metric_history(run_id, metric_key)
        if rows:
            return pd.DataFrame(rows)
    return pd.DataFrame(columns=["step", "value", "timestamp"])


@st.cache_data(ttl=60)
def load_run_params(
    run_id: str,
    tracking_uri: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieves the logged parameters for a specific experiment run.

    Args:
        run_id: Unique identifier for the tracking run.
        tracking_uri: Path to the tracking directory.

    Returns:
        Dict[str, Any]: Flat dictionary of parameters logged for the run.
    """
    store = _open_store(tracking_uri)
    if store is None:
        return {}
    with contextlib.suppress(Exception):
        return store.get_params(run_id)
    return {}


@st.cache_data(ttl=30)
def list_metric_keys(
    run_id: str,
    tracking_uri: Optional[str] = None,
) -> List[str]:
    """Retrieves the distinct metric keys logged for a specific run.

    Args:
        run_id: Unique identifier for the tracking run.
        tracking_uri: Path to the tracking directory.

    Returns:
        List[str]: Sorted sequence of distinct metric key strings.
    """
    store = _open_store(tracking_uri)
    if store is None:
        return []
    with contextlib.suppress(Exception):
        latest = store.get_latest_metrics(run_id)
        return sorted(latest.keys())
    return []


@st.cache_data(ttl=60)
def load_run_tags(
    run_id: str,
    tracking_uri: Optional[str] = None,
) -> Dict[str, str]:
    """Retrieves the metadata tags associated with a specific experiment run.

    Args:
        run_id: Unique identifier for the tracking run.
        tracking_uri: Path to the tracking directory.

    Returns:
        Dict[str, str]: Dictionary of tag keys and values.
    """
    store = _open_store(tracking_uri)
    if store is None:
        return {}
    with contextlib.suppress(Exception):
        return store.get_tags(run_id)
    return {}


@st.cache_data(ttl=30)
def load_run_artifacts(
    run_id: str,
    tracking_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves the logged artifacts metadata for a specific experiment run.

    Args:
        run_id: Unique identifier for the tracking run.
        tracking_uri: Path to the tracking directory.

    Returns:
        List[Dict[str, Any]]: Sequence of artifact metadata dictionaries.
    """
    store = _open_store(tracking_uri)
    if store is None:
        return []
    with contextlib.suppress(Exception):
        return store.get_artifacts(run_id)
    return []


@st.cache_data(ttl=30)
def load_run_dataset_events(
    run_id: str,
    tracking_uri: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Retrieves dataset lifecycle events associated with a specific run.

    Args:
        run_id: Unique identifier for the tracking run.
        tracking_uri: Path to the tracking directory.

    Returns:
        List[Dict[str, Any]]: Sequence of dataset event dictionaries.
    """
    store = _open_store(tracking_uri)
    if store is None:
        return []
    with contextlib.suppress(Exception):
        return store.get_dataset_events(run_id)
    return []


# ---------------------------------------------------------------------------
# MLflow queries (optional — requires ``mlflow`` package)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def load_mlflow_runs(
    tracking_uri: str = "mlruns",
    experiment_name: Optional[str] = None,
) -> Union[pd.DataFrame, List[Run]]:
    """Retrieves a summary of experiment runs from an MLflow tracking server.

    Args:
        tracking_uri: MLflow tracking server URI. Defaults to "mlruns".
        experiment_name: Optional experiment filter to narrow down results.

    Returns:
        Union[pd.DataFrame, List[Run]]: DataFrame containing run metadata.
    """
    if mlflow is None:
        return pd.DataFrame()

    with contextlib.suppress(Exception):
        mlflow.set_tracking_uri(tracking_uri)
        if experiment_name:
            return mlflow.search_runs(
                experiment_names=[experiment_name],
                output_format="pandas",
            )
        return mlflow.search_runs(output_format="pandas")
    return pd.DataFrame()


@st.cache_data(ttl=30)
def load_mlflow_metric_history(
    run_id: str,
    metric_key: str,
    tracking_uri: str = "mlruns",
) -> pd.DataFrame:
    """Retrieves the step-indexed history of a metric from MLflow.

    Args:
        run_id: Unique MLflow identifier for the run.
        metric_key: identifier for the metric to fetch.
        tracking_uri: MLflow tracking server URI. Defaults to "mlruns".

    Returns:
        pd.DataFrame: History with columns [step, value, timestamp].
    """
    if MlflowClient is Any:
        return pd.DataFrame(columns=["step", "value", "timestamp"])

    with contextlib.suppress(Exception):
        client = MlflowClient(tracking_uri)
        history = client.get_metric_history(run_id, metric_key)
        if history:
            return pd.DataFrame([{"step": m.step, "value": m.value, "timestamp": m.timestamp} for m in history])
    return pd.DataFrame(columns=["step", "value", "timestamp"])


@st.cache_data(ttl=30)
def list_mlflow_metric_keys(
    run_id: str,
    tracking_uri: str = "mlruns",
) -> List[str]:
    """Retrieves a list of all metric keys logged for an MLflow run.

    Args:
        run_id: Unique MLflow identifier for the run.
        tracking_uri: MLflow tracking server URI. Defaults to "mlruns".

    Returns:
        List[str]: Sorted sequence of metric key strings.
    """
    if MlflowClient is Any:
        return []

    with contextlib.suppress(Exception):
        client = MlflowClient(tracking_uri)
        run = client.get_run(run_id)
        return sorted(run.data.metrics.keys())
    return []


# ---------------------------------------------------------------------------
# ZenML queries (optional — requires ``zenml`` package)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=30)
def load_zenml_pipeline_runs() -> List[Dict[str, Any]]:
    """Retrieves the metadata for recent ZenML pipeline executions.

    Returns:
        List[Dict[str, Any]]: Sequence of pipeline run metadata dicts.
    """
    if ZenMLClient is Any:
        return []

    with contextlib.suppress(Exception):
        client = ZenMLClient()
        runs = client.list_pipeline_runs(sort_by="desc:created", size=50)
        result: List[Dict[str, Any]] = []
        for r in runs:
            result.append(
                {
                    "id": str(r.id),
                    "pipeline": r.pipeline.name if r.pipeline else "—",
                    "status": str(r.status),
                    "created": str(r.created),
                    "updated": str(r.updated),
                    "stack": r.stack.name if r.stack else "—",
                }
            )
        return result
    return []


@st.cache_data(ttl=30)
def load_zenml_run_steps(run_id: str) -> List[Dict[str, Any]]:
    """Retrieves step-level granular metadata for a ZenML pipeline run.

    Args:
        run_id: Unique ZenML identifier for the pipeline run.

    Returns:
        List[Dict[str, Any]]: Sequence of step metadata dictionaries.
    """
    if ZenMLClient is Any:
        return []

    with contextlib.suppress(Exception):
        client = ZenMLClient()
        run = client.get_pipeline_run(run_id)
        result: List[Dict[str, Any]] = []
        for step_name, step in run.steps.items():
            result.append(
                {
                    "name": step_name,
                    "status": str(step.status),
                    "created": str(step.created),
                    "updated": str(step.updated),
                }
            )
        return result
    return []
