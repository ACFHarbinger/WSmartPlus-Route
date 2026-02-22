"""
Section renderers for the Simulation Summary page.

Extracted from ``simulation_summary.py`` to keep module sizes under 400 LoC.
Functions are re-exported from the original module for backward compatibility.
"""

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.src.ui.components.charts import (
    PLOTLY_LAYOUT_DEFAULTS,
    calculate_pareto_front,
    create_pareto_scatter_chart,
)
from logic.src.ui.styles.kpi import create_kpi_row, format_number

# Metrics present in the summary JSONs (order for display)
_DISPLAY_METRICS = [
    "profit",
    "cost",
    "kg",
    "km",
    "kg/km",
    "overflows",
    "ncol",
    "kg_lost",
    "days",
    "time",
]


def _filter_by_dist(df: pd.DataFrame, dist_filter: str) -> pd.DataFrame:
    """Apply distribution filter to a DataFrame. Returns filtered copy."""
    if dist_filter == "All":
        return df
    return pd.DataFrame(df.loc[df["Distribution"] == dist_filter]).reset_index(drop=True)


def _render_kpi_overview(summary_df: pd.DataFrame, dist_filter: str) -> None:
    """Render top-level KPI cards with best values across filtered policies."""
    df = _filter_by_dist(summary_df, dist_filter)
    if df.empty:
        return

    kpi_data: Dict[str, str] = {}

    kpi_data["Policies"] = str(df["Policy"].nunique())

    if dist_filter != "All":
        kpi_data["Distribution"] = dist_filter

    if "profit_mean" in df.columns:
        best = df.loc[df["profit_mean"].idxmax()]
        kpi_data["Best Profit"] = f"{format_number(float(best['profit_mean']))} ({best['Policy']})"

    if "overflows_mean" in df.columns:
        best = df.loc[df["overflows_mean"].idxmin()]
        kpi_data["Fewest Overflows"] = f"{format_number(float(best['overflows_mean']), 1)} ({best['Policy']})"

    if "kg/km_mean" in df.columns:
        best = df.loc[df["kg/km_mean"].idxmax()]
        kpi_data["Best Efficiency"] = f"{format_number(float(best['kg/km_mean']))} kg/km ({best['Policy']})"

    if kpi_data:
        st.markdown(create_kpi_row(kpi_data), unsafe_allow_html=True)


def _render_summary_table(summary_df: pd.DataFrame, dist_filter: str) -> None:
    """Render the policy comparison table with mean +/- std."""
    st.subheader("Policy Comparison")

    df = _filter_by_dist(summary_df, dist_filter)

    if df.empty:
        st.warning("No data for selected distribution.")
        return

    # Build display DataFrame with "mean +/- std" formatted columns
    display_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        display: Dict[str, Any] = {
            "Policy": row["Policy"],
            "Distribution": row["Distribution"],
        }
        for m in _DISPLAY_METRICS:
            mean_col = f"{m}_mean"
            std_col = f"{m}_std"
            if mean_col in row.index:
                mean_val = row[mean_col]
                std_val = row.get(std_col, 0.0)
                if pd.notna(mean_val):
                    if std_val and std_val > 0:
                        display[m] = f"{mean_val:.2f} +/- {std_val:.2f}"
                    else:
                        display[m] = f"{mean_val:.2f}"
        display_rows.append(display)

    display_df = pd.DataFrame(display_rows)
    st.dataframe(display_df, width="stretch", hide_index=True)

    csv = display_df.to_csv(index=False)
    st.download_button("Download as CSV", csv, file_name="policy_summary.csv", mime="text/csv")


def _render_metric_bar_chart(summary_df: pd.DataFrame, dist_filter: str) -> None:
    """Render a bar chart ranking policies by a selected metric."""
    st.subheader("Metric Ranking")

    df = _filter_by_dist(summary_df, dist_filter)

    available_metrics = [m for m in _DISPLAY_METRICS if f"{m}_mean" in df.columns]
    if not available_metrics:
        st.info("No metrics available.")
        return

    col1, col2 = st.columns([2, 1])
    with col1:
        metric = st.selectbox("Metric", options=available_metrics, index=0, key="ss_bar_metric")
    with col2:
        sort_desc = st.checkbox("Sort descending", value=True, key="ss_bar_sort")

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    plot_df = df[["Policy", "Distribution", mean_col]].copy()
    if std_col in df.columns:
        plot_df[std_col] = df[std_col]
    else:
        plot_df[std_col] = 0.0

    plot_df = plot_df.sort_values(mean_col, ascending=not sort_desc).reset_index(drop=True)

    labels = [f"{r['Policy']} [{r['Distribution']}]" for _, r in plot_df.iterrows()]

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=labels,
            y=plot_df[mean_col],
            error_y=dict(type="data", array=plot_df[std_col], visible=True),
            marker_color=px.colors.qualitative.Set2[: len(labels)],
            hovertemplate=f"{metric}: %{{y:.2f}} +/- %{{error_y.array:.2f}}<extra>%{{x}}</extra>",
        )
    )
    fig.update_layout(
        title=f"{metric} by Policy",
        xaxis_title="Policy",
        yaxis_title=metric,
        height=450,
        xaxis_tickangle=-45,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")


def _render_pareto(summary_df: pd.DataFrame, dist_filter: str) -> None:
    """Render Pareto front scatter for two user-selected metrics."""
    st.subheader("Pareto Front Analysis")

    df = _filter_by_dist(summary_df, dist_filter)

    available_metrics = [m for m in _DISPLAY_METRICS if f"{m}_mean" in df.columns]
    if len(available_metrics) < 2:
        st.info("Need at least 2 metrics for Pareto analysis.")
        return

    col1, col2 = st.columns(2)
    with col1:
        x_default = available_metrics.index("km") if "km" in available_metrics else 0
        x_metric = st.selectbox("X-Axis Metric", options=available_metrics, index=x_default, key="ss_pareto_x")
    with col2:
        y_default = (
            available_metrics.index("profit") if "profit" in available_metrics else min(1, len(available_metrics) - 1)
        )
        y_metric = st.selectbox("Y-Axis Metric", options=available_metrics, index=y_default, key="ss_pareto_y")

    x_col = f"{x_metric}_mean"
    y_col = f"{y_metric}_mean"

    if x_col not in df.columns or y_col not in df.columns:
        st.error("Selected metrics not found in data.")
        return

    x_vals = df[x_col].tolist()
    y_vals = df[y_col].tolist()

    pareto_indices = calculate_pareto_front(x_vals, y_vals)

    color_series = df["Distribution"] if df["Distribution"].nunique() > 1 else None

    fig = create_pareto_scatter_chart(
        x=df[x_col],
        y=df[y_col],
        x_label=x_metric,
        y_label=y_metric,
        pareto_indices=pareto_indices,
        color_by=color_series,
        title=f"Pareto Front: {y_metric} vs {x_metric} (min X, max Y)",
    )

    # Add policy name annotations for Pareto points
    for idx in pareto_indices:
        row = df.iloc[idx]
        fig.add_annotation(
            x=row[x_col],
            y=row[y_col],
            text=row["Policy"],
            showarrow=True,
            arrowhead=2,
            ax=20,
            ay=-25,
            font=dict(size=10),
        )

    st.plotly_chart(fig, width="stretch")

    # Show Pareto-optimal policies
    if pareto_indices:
        pareto_df = df.iloc[pareto_indices][["Policy", "Distribution", x_col, y_col]].reset_index(drop=True)
        pareto_df.columns = pd.Index(["Policy", "Distribution", x_metric, y_metric])
        st.markdown("**Pareto-optimal policies:**")
        st.dataframe(pareto_df, width="stretch", hide_index=True)


def _render_distribution_comparison(summary_df: pd.DataFrame) -> None:
    """Render side-by-side comparison of the same policy across distributions."""
    st.subheader("Distribution Comparison")

    distributions = sorted(summary_df["Distribution"].unique().tolist())
    if len(distributions) < 2:
        st.info("Need at least 2 distributions for comparison. Only found: " + ", ".join(distributions))
        return

    available_metrics = [m for m in _DISPLAY_METRICS if f"{m}_mean" in summary_df.columns]
    if not available_metrics:
        st.info("No metrics available.")
        return

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Metric", options=available_metrics, index=0, key="ss_distcomp_metric")
    with col2:
        selected_dists = st.multiselect(
            "Distributions",
            options=distributions,
            default=distributions,
            key="ss_distcomp_dists",
        )

    if not selected_dists:
        st.warning("Select at least one distribution.")
        return

    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    # Filter to selected distributions
    df = pd.DataFrame(summary_df.loc[summary_df["Distribution"].isin(selected_dists)]).reset_index(drop=True)

    # Find policies that appear in at least one selected distribution
    policies = sorted(df["Policy"].unique().tolist())

    # Build grouped bar chart: X = policy, grouped by distribution
    fig = go.Figure()
    colors = px.colors.qualitative.Set2

    for i, dist in enumerate(selected_dists):
        dist_df = df[df["Distribution"] == dist]
        # Align to full policy list (NaN for missing)
        policy_vals = []
        policy_stds = []
        for p in policies:
            match = dist_df[dist_df["Policy"] == p]
            if not match.empty:
                policy_vals.append(float(match.iloc[0][mean_col]))
                policy_stds.append(float(match.iloc[0].get(std_col, 0.0)) if std_col in match.columns else 0.0)
            else:
                policy_vals.append(float("nan"))
                policy_stds.append(0.0)

        fig.add_trace(
            go.Bar(
                x=policies,
                y=policy_vals,
                name=dist,
                marker_color=colors[i % len(colors)],
                error_y=dict(type="data", array=policy_stds, visible=True),
                hovertemplate=f"{dist}<br>{metric}: %{{y:.2f}}<extra>%{{x}}</extra>",
            )
        )

    fig.update_layout(
        barmode="group",
        title=f"{metric} by Policy — Grouped by Distribution",
        xaxis_title="Policy",
        yaxis_title=metric,
        height=500,
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")

    # Pivot table: rows=policies, columns=distributions
    st.markdown("**Summary Table**")
    pivot_rows: List[Dict[str, Any]] = []
    for p in policies:
        row: Dict[str, Any] = {"Policy": p}
        for dist in selected_dists:
            match = df[(df["Policy"] == p) & (df["Distribution"] == dist)]
            if not match.empty:
                mean_val = float(match.iloc[0][mean_col])
                std_val = float(match.iloc[0].get(std_col, 0.0)) if std_col in match.columns else 0.0
                if std_val > 0:
                    row[dist] = f"{mean_val:.2f} +/- {std_val:.2f}"
                else:
                    row[dist] = f"{mean_val:.2f}"
            else:
                row[dist] = "—"
        pivot_rows.append(row)

    pivot_df = pd.DataFrame(pivot_rows)
    st.dataframe(pivot_df, width="stretch", hide_index=True)


def _render_daily_timeseries(daily_df: pd.DataFrame, dist_filter: str) -> None:
    """Render daily metric time-series overlaid per policy."""
    st.subheader("Daily Time-Series")

    if daily_df.empty:
        st.info("No daily data loaded.")
        return

    df = _filter_by_dist(daily_df, dist_filter)
    if df.empty:
        st.warning(f"No daily data for distribution: {dist_filter}")
        return

    # Exclude non-metric columns
    metric_cols = [c for c in df.columns if c not in ("Policy", "Distribution", "day", "tour")]
    if not metric_cols:
        st.info("No metrics found in daily data.")
        return

    metric = st.selectbox("Metric", options=metric_cols, index=0, key="ss_daily_metric")

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    policies = df["Policy"].unique()

    for i, policy in enumerate(policies):
        mask = df["Policy"] == policy
        policy_df = pd.DataFrame(df.loc[mask]).sort_values("day")

        # Aggregate across distributions if dist_filter is All
        if dist_filter == "All":
            grouped = policy_df.groupby("day")[metric].agg(["mean", "std"]).reset_index()
            fig.add_trace(
                go.Scatter(
                    x=grouped["day"],
                    y=grouped["mean"],
                    mode="lines+markers",
                    name=policy,
                    line=dict(color=colors[i % len(colors)]),
                    marker=dict(size=4),
                    hovertemplate=f"{policy}<br>{metric}: %{{y:.2f}}<extra>Day %{{x}}</extra>",
                )
            )
        else:
            fig.add_trace(
                go.Scatter(
                    x=policy_df["day"],
                    y=policy_df[metric],
                    mode="lines+markers",
                    name=policy,
                    line=dict(color=colors[i % len(colors)]),
                    marker=dict(size=4),
                    hovertemplate=f"{policy}<br>{metric}: %{{y:.2f}}<extra>Day %{{x}}</extra>",
                )
            )

    fig.update_layout(
        title=f"Daily {metric} by Policy",
        xaxis_title="Day",
        yaxis_title=metric,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    st.plotly_chart(fig, width="stretch")
