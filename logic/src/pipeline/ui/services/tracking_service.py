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

    from logic.src.pipeline.ui.services.tracking_service import (
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
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

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
        from logic.src.tracking.core.store import TrackingStore

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
