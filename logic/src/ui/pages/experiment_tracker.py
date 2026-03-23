"""Experiment Tracker mode for the Streamlit dashboard.

Provides a unified view of all tracked experiments from three sources:

1. **WSTracker** (SQLite) — native experiment database
2. **MLflow** (optional) — remote/local tracking server
3. **ZenML** (optional) — pipeline orchestration runs & step statuses

Sections: Run Table, Run Detail, Metric Explorer, Run Comparison,
MLflow Explorer, ZenML Pipeline Runs, Artifacts.
"""

import json
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
    load_run_dataset_events,
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
    st.subheader("🏷️ Tags")
    tags = load_run_tags(run_id)
    if tags:
        st.markdown(create_kpi_row(tags), unsafe_allow_html=True)
    else:
        st.caption("No tags recorded.")

    st.markdown("---")
    st.subheader("⚙️ Parameters")
    params = load_run_params(run_id)

    if not params:
        st.caption("No parameters recorded.")
        return

    # Categorize parameters
    policy_params = {k: v for k, v in params.items() if k.startswith("policy_params/")}
    global_params = {k: v for k, v in params.items() if not k.startswith("policy_params/")}

    if policy_params:
        tab_global, tab_policy = st.tabs(["🌐 Global Params", "🔧 Policy Params"])

        with tab_global:
            _render_params_table(global_params)

        with tab_policy:
            # Group policy params hierarchically: policy -> sample -> {key: val}
            # Format: 'policy_params/{policy_name}/s{sample_id}/{key}'
            grouped: Dict[str, Dict[str, Dict[str, Any]]] = {}
            for k, v in policy_params.items():
                parts = k.split("/")
                if len(parts) >= 4:
                    p_name = parts[1]
                    s_id = parts[2]
                    key = "/".join(parts[3:])

                    if p_name not in grouped:
                        grouped[p_name] = {}
                    if s_id not in grouped[p_name]:
                        grouped[p_name][s_id] = {}
                    grouped[p_name][s_id][key] = v
                else:
                    # Fallback for unexpected formats
                    if "Other" not in grouped:
                        grouped["Other"] = {"s0": {}}
                    grouped["Other"]["s0"][k] = v

            for p_name, samples in sorted(grouped.items()):
                st.markdown(f"#### 🏷️ Policy: `{p_name}`")
                for s_id, p_dict in sorted(samples.items()):
                    with st.expander(f"Sample `{s_id}`", expanded=True):
                        _render_params_table(p_dict)
    else:
        _render_params_table(global_params)


def _render_params_table(params: Dict[str, Any]) -> None:
    """Helper to render a clean parameters table."""
    if not params:
        st.caption("No parameters in this category.")
        return

    df_params = pd.DataFrame(
        {"Parameter": list(params.keys()), "Value": [str(v) for v in params.values()]}
    ).sort_values("Parameter")

    st.dataframe(
        df_params,
        width="stretch",
        hide_index=True,
        height=min(600, 35 + len(df_params) * 35),
    )


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

    df_artifacts = pd.DataFrame(rows)
    st.dataframe(
        df_artifacts,
        width="stretch",
        hide_index=True,
        height=min(300, 60 + len(rows) * 35),
    )


_EVENT_TYPE_ICONS = {
    "load": "📥",
    "generate": "🔧",
    "mutate": "🔀",
    "save": "💾",
    "hash_change": "#️⃣",
    "schema_change": "📋",
    "augment": "🔄",
    "regenerate": "♻️",
}


def _render_dataset_events(run_id: str) -> None:
    """Display dataset lifecycle events for a WSTracker run."""
    events = load_run_dataset_events(run_id)
    if not events:
        st.caption("No dataset events recorded for this run.")
        return

    rows = []
    for e in events:
        etype = e.get("event_type", "—")
        icon = _EVENT_TYPE_ICONS.get(etype, "❓")
        meta = e.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}

        # Source location
        src_file = meta.get("source_file", "")
        src_line = meta.get("source_line", "")
        source = f"{src_file}:{src_line}" if src_file else "—"

        rows.append(
            {
                "Event": f"{icon} {etype}",
                "Variable": meta.get("variable_name", "—"),
                "File": e.get("file_path", "—"),
                "Source": source,
                "Shape": e.get("shape", "—"),
                "Size": _fmt_size(e.get("size_bytes")),
                "Timestamp": e.get("timestamp", "—"),
            }
        )

    df_events = pd.DataFrame(rows)
    total_events = len(df_events)

    # --- Dataset Event Statistics ---
    st.markdown(f"##### Event Summary ({total_events:,} total)")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.caption("Event Types")
        type_counts = df_events["Event"].value_counts().reset_index()
        type_counts.columns = ["Event", "Count"]  # type: ignore[assignment]
        st.dataframe(type_counts, use_container_width=True, hide_index=True)

    with c2:
        st.caption("Top Variables")
        vars_df = df_events[df_events["Variable"] != "—"]
        if not vars_df.empty:
            var_counts = vars_df["Variable"].value_counts().head(5).reset_index()
            var_counts.columns = ["Variable", "Count"]  # type: ignore[assignment]
            st.dataframe(var_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No named variables logged")

    with c3:
        st.caption("Top Source Files")
        # Extract just the filename from the Source string "path/file.py:123" if available
        # or fallback to "File" column if "Source" isn't present.
        valid_sources = df_events[df_events["Source"] != "—"]
        if not valid_sources.empty:
            src_counts = (
                valid_sources["Source"]
                .apply(lambda s: "/".join(s.split(":")[0].split("/")[-2:]))
                .value_counts()
                .head(5)
                .reset_index()
            )
            src_counts.columns = ["Source", "Count"]  # type: ignore[assignment]
            st.dataframe(src_counts, use_container_width=True, hide_index=True)
        else:
            st.info("No source locations logged")

    st.markdown("##### Detailed Event Timeline")
    if total_events > 5000:
        st.warning(f"Showing first 5,000 of {total_events:,} events to maintain performance.")
        df_display = df_events.head(5000)
    else:
        df_display = df_events

    st.dataframe(
        df_display,
        width="stretch",
        hide_index=True,
        height=min(400, 60 + len(df_display) * 35),
    )


def _fmt_size(size_bytes: Any) -> str:
    """Format byte count into a human-readable string."""
    if size_bytes is None:
        return "—"
    try:
        b = int(size_bytes)
    except (TypeError, ValueError):
        return "—"
    for unit in ("B", "KB", "MB", "GB"):
        if abs(b) < 1024:
            return f"{b:.0f} {unit}" if unit == "B" else f"{b:.1f} {unit}"
        b /= 1024.0
    return f"{b:.1f} TB"


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

                # 5. Dataset Events
                st.markdown("---")
                st.subheader("📂 Dataset Events")
                _render_dataset_events(selected_run_id)

            # 6. Run Comparison (always available)
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
