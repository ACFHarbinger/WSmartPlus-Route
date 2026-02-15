from typing import Any, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from logic.src.pipeline.ui.components.charts import (
    create_radar_chart,
    create_simulation_metrics_chart,
)
from logic.src.pipeline.ui.services.data_loader import compute_daily_stats, compute_summary_statistics
from logic.src.pipeline.ui.services.log_parser import filter_entries, get_unique_policies


def render_policy_comparison(entries: List[Any], selected_day: int) -> None:
    """Render radar chart comparing all policies for the selected day."""
    policies = get_unique_policies(entries)
    if len(policies) < 2:
        return

    st.subheader("Policy Comparison")
    radar_metrics = ["profit", "km", "kg", "overflows", "cost", "kg/km"]

    policy_metrics: Dict[str, Dict[str, float]] = {}
    for policy in policies:
        day_entries = filter_entries(entries, policy=policy, day=selected_day)
        if not day_entries:
            continue

        metrics: Dict[str, float] = {}
        for metric in radar_metrics:
            values = [e.data.get(metric, 0) for e in day_entries if metric in e.data]
            if values:
                metrics[metric] = float(np.mean(values))
        if metrics:
            policy_metrics[policy] = metrics

    if len(policy_metrics) < 2:
        return

    fig = create_radar_chart(policy_metrics, radar_metrics)
    st.plotly_chart(fig, use_container_width=True)


def render_summary_statistics(entries: List[Any], controls: Dict[str, Any]) -> None:
    """Render descriptive statistics table."""
    summary = compute_summary_statistics(entries, policy=controls["selected_policy"])
    if not summary:
        return

    st.subheader("Summary Statistics")
    rows = []
    for metric, stats in summary.items():
        rows.append(
            {
                "Metric": metric,
                "Mean": round(stats["mean"], 2),
                "Std": round(stats["std"], 2),
                "Min": round(stats["min"], 2),
                "Max": round(stats["max"], 2),
                "Total": round(stats["total"], 2),
            }
        )

    if rows:
        df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(df, use_container_width=True)


def render_metric_charts(entries: List[Any], controls: Dict[str, Any]) -> None:
    """Render evaluation metrics charts with user-selectable metrics."""
    df = compute_daily_stats(entries, policy=controls["selected_policy"])

    if not df.empty:
        all_metrics = ["profit", "km", "kg", "overflows", "ncol", "kg_lost", "kg/km", "cost"]
        available_plot_metrics = [m for m in all_metrics if f"{m}_mean" in df.columns]

        if available_plot_metrics:
            selected_metrics = st.multiselect(
                "Select metrics to plot",
                options=available_plot_metrics,
                default=available_plot_metrics[:3],
                key="metrics_select_multi",
            )

            if selected_metrics:
                fig = create_simulation_metrics_chart(df=df, metrics=selected_metrics, show_std=True)
                st.plotly_chart(fig, use_container_width=True)

        csv = df.to_csv(index=False)
        st.download_button(
            "Download Daily Stats CSV",
            csv,
            file_name="daily_stats.csv",
            mime="text/csv",
            key="download_daily_stats_btn",
        )
