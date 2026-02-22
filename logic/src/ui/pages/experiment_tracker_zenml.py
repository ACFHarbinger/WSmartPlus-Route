"""
ZenML pipeline runs section for the Experiment Tracker page.

Extracted from ``experiment_tracker.py`` to keep module sizes under 400 LoC.
"""

from typing import Dict

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from logic.src.ui.components.charts import PLOTLY_LAYOUT_DEFAULTS
from logic.src.ui.services.tracking_service import (
    load_zenml_pipeline_runs,
    load_zenml_run_steps,
)

_STATUS_ICONS = {
    "completed": "✅",
    "running": "🔄",
    "failed": "❌",
    "interrupted": "⚠️",
    "cached": "💾",
}


def _render_zenml_pipelines() -> None:
    """ZenML pipeline runs with step status drill-down."""
    st.subheader("🚀 ZenML Pipeline Runs")

    zenml_runs = load_zenml_pipeline_runs()

    if not zenml_runs:
        st.info(
            "No ZenML pipeline runs found.\n\n"
            "Enable ZenML tracking (`tracking.zenml_enabled: true`) and "
            "run a training, evaluation, or simulation to populate this view."
        )
        return

    # Pipeline run table
    df = pd.DataFrame(zenml_runs)
    df["status"] = df["status"].apply(lambda s: f"{_STATUS_ICONS.get(s, '❓')} {s}")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        height=min(400, 60 + len(df) * 35),
    )

    # Step drill-down
    run_options = {f"{r['id'][:8]}  {r['pipeline']}": r["id"] for r in zenml_runs}

    selected_zenml_label = st.selectbox(
        "Select ZenML run to inspect steps",
        options=list(run_options.keys()),
        index=0,
        key="zenml_run_select",
    )

    selected_zenml_id = run_options.get(selected_zenml_label)
    if not selected_zenml_id:
        return

    steps = load_zenml_run_steps(selected_zenml_id)
    if not steps:
        st.caption("No step data available for this pipeline run.")
        return

    st.markdown("**Pipeline Steps**")

    # Step status timeline
    step_df = pd.DataFrame(steps)
    step_df["status"] = step_df["status"].apply(lambda s: f"{_STATUS_ICONS.get(s, '❓')} {s}")

    st.dataframe(
        step_df,
        use_container_width=True,
        hide_index=True,
    )

    # Step status chart (horizontal bar)
    status_counts: Dict[str, int] = {}
    for s in steps:
        raw_status = s.get("status", "unknown")
        status_counts[raw_status] = status_counts.get(raw_status, 0) + 1

    if status_counts:
        fig = go.Figure(
            data=[
                go.Bar(
                    x=list(status_counts.values()),
                    y=list(status_counts.keys()),
                    orientation="h",
                    marker_color=[
                        "#48bb78" if k == "completed" else "#fc8181" if k == "failed" else "#ecc94b"
                        for k in status_counts
                    ],
                )
            ]
        )
        fig.update_layout(
            title="Step Status Summary",
            height=250,
            xaxis_title="Count",
            yaxis_title="Status",
            **PLOTLY_LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig, use_container_width=True)
