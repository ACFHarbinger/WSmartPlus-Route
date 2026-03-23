"""WSTracker read service for the Streamlit dashboard.

Provides cached Streamlit functions to query the WSTracker SQLite
database so simulation and training metrics can be rendered alongside
the existing JSON-log-based charts.

All public functions accept an optional *tracking_uri* parameter; when
omitted the default ``assets/tracking`` directory relative to the
project root is used.

Typical usage
-------------
::

    from logic.src.ui.services.tracking_service import (
        load_tracking_runs,
        load_run_metrics,
        load_run_params,
        list_metric_keys,
    )

    runs = load_tracking_runs()
    run_id = runs[0]["id"]
    df = load_run_metrics(run_id, "gurobi/s0/profit")
    st.line_chart(df.set_index("step")["value"])
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
    mlflow = None
    Run = Any
    MlflowClient = Any

try:
    from zenml.client import Client as ZenMLClient
except ImportError:
    ZenMLClient = Any

with contextlib.suppress(ImportError):
    from logic.src.tracking.core.store import TrackingStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_tracking_uri(tracking_uri: Optional[str]) -> str:
    if tracking_uri:
        return tracking_uri
    return str(Path.cwd() / "assets" / "tracking")


def _open_store(tracking_uri: Optional[str]) -> Optional[Any]:
    """Return a :class:`TrackingStore` for *tracking_uri*, or ``None``."""
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
    """Return all runs from the WSTracker database.

    Args:
        tracking_uri: Path to the tracking directory containing
            ``tracking.db``.  Defaults to ``assets/tracking``.
        run_type: Optional filter (e.g. ``"training"`` or
            ``"simulation"``).

    Returns:
        List of run dicts with keys ``id``, ``name``, ``status``,
        ``run_type``, ``start_time``, ``end_time``.
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
    """Return the step-indexed history of *metric_key* for *run_id*.

    Args:
        run_id: UUID of the run.
        metric_key: Exact metric key string (e.g.
            ``"gurobi/s0/profit"``).
        tracking_uri: Path to the tracking directory.

    Returns:
        DataFrame with columns ``step``, ``value``, ``timestamp``.
        Empty DataFrame when the run or metric is not found.
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
    """Return the logged parameters for *run_id*.

    Args:
        run_id: UUID of the run.
        tracking_uri: Path to the tracking directory.

    Returns:
        Flat param dict, or an empty dict when not found.
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
    """Return the distinct metric keys logged for *run_id*.

    Args:
        run_id: UUID of the run.
        tracking_uri: Path to the tracking directory.

    Returns:
        Sorted list of metric key strings.
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
    """Return the tags for *run_id*.

    Args:
        run_id: UUID of the run.
        tracking_uri: Path to the tracking directory.

    Returns:
        Dict of tag key/values, or an empty dict when not found.
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
    """Return the logged artifacts for *run_id*.

    Args:
        run_id: UUID of the run.
        tracking_uri: Path to the tracking directory.

    Returns:
        List of artifact dicts (path, artifact_type, timestamp, etc.).
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
    """Return the dataset lifecycle events for *run_id*.

    Args:
        run_id: UUID of the run.
        tracking_uri: Path to the tracking directory.

    Returns:
        List of event dicts (event_type, file_path, timestamp, etc.).
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
    """Return a DataFrame of MLflow runs.

    Args:
        tracking_uri: MLflow tracking server URI.
        experiment_name: Optional experiment filter.

    Returns:
        DataFrame with columns from ``mlflow.search_runs``, or empty.
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
    """Return the step-indexed history of *metric_key* from MLflow.

    Args:
        run_id: MLflow run ID.
        metric_key: Metric key to fetch.
        tracking_uri: MLflow tracking server URI.

    Returns:
        DataFrame with columns ``step``, ``value``, ``timestamp``.
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
    """Return metric keys logged for an MLflow run.

    Args:
        run_id: MLflow run ID.
        tracking_uri: MLflow tracking server URI.

    Returns:
        Sorted list of metric key strings.
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
    """Return recent ZenML pipeline runs.

    Returns:
        List of dicts with pipeline run metadata (name, status, etc.).
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
    """Return the steps of a ZenML pipeline run.

    Args:
        run_id: UUID of the ZenML pipeline run.

    Returns:
        List of step dicts (name, status, duration, etc.).
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
