"""
MLflow explorer section for the Experiment Tracker page.

Extracted from ``experiment_tracker.py`` to keep module sizes under 400 LoC.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic.src.constants.dashboard import PLOTLY_LAYOUT_DEFAULTS
from logic.src.ui.services.tracking_service import (
    list_mlflow_metric_keys,
    load_mlflow_metric_history,
    load_mlflow_runs,
)

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


def _render_mlflow_explorer(tracking_uri: str, experiment_name: str) -> None:
    """Full MLflow runs explorer with metric charting."""
    st.subheader("🧪 MLflow Runs")

    # The type is Union[DataFrame, list[Run]]
    mlflow_df = load_mlflow_runs(tracking_uri, experiment_name)

    # Use len() instead of .empty to satisfy both list and DataFrame types
    if mlflow_df is None or len(mlflow_df) == 0:
        st.info(
            f"No MLflow runs found at `{tracking_uri}`.\n\n"
            "Enable MLflow tracking and run a training to populate this view."
        )
        return

    # To use DataFrame-specific methods (.columns, .head), cast it or narrow it

    if not isinstance(mlflow_df, pd.DataFrame):
        # Convert list to DataFrame if necessary
        mlflow_df = pd.DataFrame([r.__dict__ for r in mlflow_df])

    # Now Pyrefly knows mlflow_df is definitely a DataFrame
    display_cols = [c for c in mlflow_df.columns if not c.startswith("params.")]
    display_cols = [c for c in display_cols if c in mlflow_df.columns][:10]

    st.dataframe(
        mlflow_df[display_cols] if display_cols else mlflow_df.head(20),
        width="stretch",
        hide_index=True,
        height=min(400, 60 + len(mlflow_df) * 35),
    )

    # Select a run for metric exploration
    run_ids = mlflow_df["run_id"].tolist() if "run_id" in mlflow_df.columns else []
    if not run_ids:
        return

    selected_mlflow_run = st.selectbox(
        "Select MLflow run to explore metrics",
        options=run_ids,
        format_func=lambda x: x[:8],
        key="mlflow_run_select",
    )

    if not selected_mlflow_run:
        return

    # Metric keys
    metric_keys = list_mlflow_metric_keys(selected_mlflow_run, tracking_uri)
    if not metric_keys:
        st.caption("No metric history available for this run.")
        return

    selected_mf_keys = st.multiselect(
        "Select MLflow metrics to chart",
        options=metric_keys,
        default=metric_keys[: min(3, len(metric_keys))],
        key="mlflow_metric_select",
    )

    if not selected_mf_keys:
        return

    fig = go.Figure()
    for i, key in enumerate(selected_mf_keys):
        df = load_mlflow_metric_history(selected_mlflow_run, key, tracking_uri)
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
        title="MLflow Metric History",
        xaxis_title="Step",
        yaxis_title="Value",
        height=400,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")

    # Params table
    run_row = mlflow_df[mlflow_df["run_id"] == selected_mlflow_run]
    param_cols = [c for c in run_row.columns if c.startswith("params.")]
    if param_cols:
        with st.expander("MLflow Parameters", expanded=False):
            param_data = {c.replace("params.", ""): run_row[c].iloc[0] for c in param_cols}
            st.json(param_data)
