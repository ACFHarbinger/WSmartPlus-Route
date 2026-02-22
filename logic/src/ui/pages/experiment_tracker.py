"""Experiment Tracker mode for the Streamlit dashboard.

Provides a unified view of all tracked experiments from three sources:

1. **WSTracker** (SQLite) — native experiment database
2. **MLflow** (optional) — remote/local tracking server
3. **ZenML** (optional) — pipeline orchestration runs & step statuses

Sections: Run Table, Run Detail, Metric Explorer, Run Comparison,
MLflow Explorer, ZenML Pipeline Runs, Artifacts.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic.src.ui.components.charts import PLOTLY_LAYOUT_DEFAULTS
from logic.src.ui.pages.experiment_tracker_mlflow import _render_mlflow_explorer
from logic.src.ui.pages.experiment_tracker_zenml import _render_zenml_pipelines
from logic.src.ui.services.tracking_service import (
    list_metric_keys,
    load_run_artifacts,
    load_run_metrics,
    load_run_params,
    load_run_tags,
    load_tracking_runs,
)
from logic.src.ui.styles.kpi import create_kpi_row

# ---------------------------------------------------------------------------
# Colour palette for multi-run overlays
# ---------------------------------------------------------------------------

_PALETTE = [
    "#667eea",
    "#f093fb",
    "#4fd1c5",
    "#f6ad55",
    "#fc8181",
    "#90cdf4",
    "#9ae6b4",
    "#fbd38d",
    "#d6bcfa",
    "#fed7d7",
]

_STATUS_ICONS = {
    "completed": "✅",
    "running": "🔄",
    "failed": "❌",
    "interrupted": "⚠️",
    "cached": "💾",
}

# ---------------------------------------------------------------------------
# WSTracker sections
# ---------------------------------------------------------------------------


def _render_run_table(
    runs: List[Dict[str, Any]],
    run_type_filter: Optional[str],
) -> Optional[str]:
    """Render a filterable table of WSTracker runs; return selected run_id."""
    if run_type_filter and run_type_filter != "All":
        runs = [r for r in runs if r.get("run_type") == run_type_filter]

    if not runs:
        st.info("No WSTracker runs found. Start a tracked run to see data here.")
        return None

    rows = []
    for r in runs:
        rows.append(
            {
                "ID": r.get("id", "")[:8],
                "Full ID": r.get("id", ""),
                "Experiment": r.get("experiment_name", r.get("name", "—")),
                "Type": r.get("run_type", "—"),
                "Status": r.get("status", "—"),
                "Started": r.get("start_time", ""),
                "Ended": r.get("end_time", ""),
            }
        )

    df = pd.DataFrame(rows)
    df["Status"] = df["Status"].apply(lambda s: f"{_STATUS_ICONS.get(s, '❓')} {s}")

    st.dataframe(
        df.drop(columns=["Full ID"]),
        width="stretch",
        height=min(400, 60 + len(df) * 35),
        hide_index=True,
    )

    run_options = {f"{r['ID']}  {r['Experiment']}": r["Full ID"] for r in rows}
    if not run_options:
        return None

    selected_label = st.selectbox(
        "Select a run to inspect",
        options=list(run_options.keys()),
        index=0,
    )
    return run_options.get(selected_label)


def _render_run_detail(run_id: str) -> None:
    """Display params and tags for a single WSTracker run."""
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏷️ Tags")
        tags = load_run_tags(run_id)
        if tags:
            st.markdown(create_kpi_row(tags), unsafe_allow_html=True)
        else:
            st.caption("No tags recorded.")

    with col2:
        st.subheader("⚙️ Parameters")
        params = load_run_params(run_id)
        if params:
            with st.expander("View all parameters", expanded=False):
                st.json(params)

            key_params = {k: v for k, v in list(params.items())[:12]}
            if key_params:
                st.dataframe(
                    pd.DataFrame({"param": key_params.keys(), "value": [str(v) for v in key_params.values()]}),
                    width="stretch",
                    hide_index=True,
                    height=min(300, 35 + len(key_params) * 35),
                )
        else:
            st.caption("No parameters recorded.")


def _render_metric_explorer(run_id: str) -> None:
    """Interactive metric explorer for a single WSTracker run."""
    keys = list_metric_keys(run_id)
    if not keys:
        st.info("No metrics have been logged for this run yet.")
        return

    selected_keys = st.multiselect(
        "Select metrics to plot",
        options=keys,
        default=keys[: min(3, len(keys))],
        help="Choose one or more metric keys to visualize",
        key="wst_metric_select",
    )

    if not selected_keys:
        return

    fig = go.Figure()
    for i, key in enumerate(selected_keys):
        df = load_run_metrics(run_id, key)
        if df.empty:
            continue

        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["value"],
                mode="lines+markers",
                name=key,
                marker=dict(size=4, color=color),
                line=dict(color=color),
                hovertemplate=f"<b>{key}</b><br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

    fig.update_layout(
        xaxis_title="Step",
        yaxis_title="Value",
        height=450,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")

    summary: Dict[str, Any] = {}
    for key in selected_keys:
        df = load_run_metrics(run_id, key)
        if not df.empty:
            summary[key] = round(float(df["value"].iloc[-1]), 4)

    if summary:
        st.markdown(create_kpi_row(summary), unsafe_allow_html=True)


def _render_run_comparison(runs: List[Dict[str, Any]]) -> None:
    """Compare WSTracker metrics across 2+ selected runs."""
    if len(runs) < 2:
        st.info("Need at least 2 WSTracker runs to compare.")
        return

    run_labels = {
        f"{r.get('id', '')[:8]}  {r.get('experiment_name', r.get('name', '—'))}": r.get("id", "") for r in runs
    }

    selected_labels = st.multiselect(
        "Select runs to compare",
        options=list(run_labels.keys()),
        default=list(run_labels.keys())[: min(3, len(run_labels))],
        key="wst_compare_runs",
    )

    if len(selected_labels) < 2:
        return

    selected_ids = [run_labels[label] for label in selected_labels]

    all_keys_per_run = {rid: set(list_metric_keys(rid)) for rid in selected_ids}
    shared_keys = sorted(set.intersection(*all_keys_per_run.values())) if all_keys_per_run else []

    if not shared_keys:
        st.warning("No shared metric keys found across the selected runs.")
        return

    comparison_key = st.selectbox(
        "Metric to compare",
        options=shared_keys,
        index=0,
        key="wst_compare_metric",
    )

    if not comparison_key:
        return

    fig = go.Figure()
    comparison_table: List[Dict[str, Any]] = []

    for i, (label, rid) in enumerate(zip(selected_labels, selected_ids)):
        df = load_run_metrics(rid, comparison_key)
        if df.empty:
            continue

        color = _PALETTE[i % len(_PALETTE)]
        short_label = label[:30]

        fig.add_trace(
            go.Scatter(
                x=df["step"],
                y=df["value"],
                mode="lines",
                name=short_label,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>{short_label}</b><br>Step: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
            )
        )

        values = df["value"]
        comparison_table.append(
            {
                "Run": short_label,
                "Latest": round(float(values.iloc[-1]), 4),
                "Best": round(float(values.min()), 4),
                "Mean": round(float(values.mean()), 4),
                "Std": round(float(values.std()), 4),
                "Steps": len(values),
            }
        )

    fig.update_layout(
        title=f"Comparison: {comparison_key}",
        xaxis_title="Step",
        yaxis_title="Value",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")

    if comparison_table:
        st.dataframe(
            pd.DataFrame(comparison_table),
            width="stretch",
            hide_index=True,
        )


def _render_artifacts(run_id: str) -> None:
    """Display logged artifacts for a WSTracker run."""
    artifacts = load_run_artifacts(run_id)
    if not artifacts:
        st.caption("No artifacts logged for this run.")
        return

    rows = []
    for a in artifacts:
        rows.append(
            {
                "Path": a.get("path", "—"),
                "Type": a.get("artifact_type", "—"),
                "Logged At": a.get("timestamp", "—"),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        width="stretch",
        hide_index=True,
        height=min(300, 60 + len(rows) * 35),
    )


# MLflow and ZenML sections in experiment_tracker_mlflow.py and experiment_tracker_zenml.py


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def _render_tracker_sidebar(
    run_types: List[str],
) -> Dict[str, Any]:
    """Render sidebar controls for the experiment tracker."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔬 Experiment Tracker")

    run_type_filter = st.sidebar.selectbox(
        "Filter by Run Type",
        options=["All"] + run_types,
        index=0,
        help="Filter WSTracker runs by type",
    )

    # MLflow settings
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧪 MLflow")
    mlflow_uri = st.sidebar.text_input(
        "MLflow Tracking URI",
        value="mlruns",
        help="Path or URL to the MLflow tracking server",
    )
    mlflow_experiment = st.sidebar.text_input(
        "MLflow Experiment",
        value="wsmart-route",
        help="Filter MLflow runs by experiment name (leave blank for all)",
    )

    return {
        "run_type_filter": run_type_filter,
        "mlflow_uri": mlflow_uri,
        "mlflow_experiment": mlflow_experiment if mlflow_experiment else None,
    }


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


def render_experiment_tracker() -> None:
    """Render the Experiment Tracker mode."""
    st.title("🔬 Experiment Tracker")
    st.markdown(
        "Unified view of experiments from **WSTracker**, **MLflow**, and **ZenML**. "
        "Browse runs, inspect metrics, compare results, and view pipeline statuses."
    )

    # Load WSTracker runs
    all_runs = load_tracking_runs()
    run_types = sorted({r.get("run_type", "") for r in all_runs if r.get("run_type")})

    # Sidebar controls
    controls = _render_tracker_sidebar(run_types)

    # ====================================================================
    # Tab layout: WSTracker | MLflow | ZenML
    # ====================================================================
    tab_wst, tab_mlflow, tab_zenml = st.tabs(["📊 WSTracker", "🧪 MLflow", "🚀 ZenML"])

    # --- WSTracker tab ---
    with tab_wst:
        if not all_runs:
            st.info(
                "No tracked runs found in the WSTracker database.\n\n"
                "Run a training, evaluation, or simulation with tracking "
                "enabled to see data here."
            )
        else:
            # 1. Run Table
            selected_run_id = _render_run_table(all_runs, controls["run_type_filter"])

            if selected_run_id:
                # 2. Run Detail
                st.markdown("---")
                _render_run_detail(selected_run_id)

                # 3. Metric Explorer
                st.markdown("---")
                st.subheader("📈 Metric Explorer")
                _render_metric_explorer(selected_run_id)

                # 4. Artifacts
                st.markdown("---")
                st.subheader("📦 Artifacts")
                _render_artifacts(selected_run_id)

            # 5. Run Comparison (always available)
            st.markdown("---")
            st.subheader("🔀 Run Comparison")
            _render_run_comparison(all_runs)

    # --- MLflow tab ---
    with tab_mlflow:
        _render_mlflow_explorer(
            controls["mlflow_uri"],
            controls["mlflow_experiment"],
        )

    # --- ZenML tab ---
    with tab_zenml:
        _render_zenml_pipelines()
