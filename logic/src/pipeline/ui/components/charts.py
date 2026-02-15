# Copyright (c) WSmart-Route. All rights reserved.
"""
Plotly chart generators for the dashboard.

Provides reusable chart components for training and simulation visualization.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Shared Plotly layout defaults for visual consistency across all charts
PLOTLY_LAYOUT_DEFAULTS: Dict[str, Any] = {
    "font_family": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
    "plot_bgcolor": "rgba(0,0,0,0)",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "margin": dict(l=40, r=20, t=50, b=40),
    "xaxis": dict(gridcolor="#e8eaed", gridwidth=1, zeroline=False),
    "yaxis": dict(gridcolor="#e8eaed", gridwidth=1, zeroline=False),
}


def apply_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Apply moving average smoothing to a series."""
    if window <= 1:
        return data
    return data.rolling(window=window, min_periods=1).mean()


def create_training_loss_chart(
    runs_data: Dict[str, pd.DataFrame],
    metric_y1: str = "train_loss",
    metric_y2: Optional[str] = "val_cost",
    x_axis: str = "epoch",
    smoothing: int = 1,
) -> go.Figure:
    """
    Create a multi-run comparison chart with optional dual Y-axes.

    Args:
        runs_data: Dict mapping run name to DataFrame.
        metric_y1: Primary metric for left Y-axis.
        metric_y2: Optional secondary metric for right Y-axis.
        x_axis: X-axis column name (epoch or step).
        smoothing: Moving average window size.

    Returns:
        Plotly Figure object.
    """
    has_secondary = metric_y2 is not None

    fig = make_subplots(
        specs=[[{"secondary_y": has_secondary}]],
        subplot_titles=[f"Training Metrics ({x_axis})"],
    )

    colors = px.colors.qualitative.Set2

    for i, (run_name, df) in enumerate(runs_data.items()):
        if df.empty:
            continue

        color = colors[i % len(colors)]

        # Primary metric
        if metric_y1 in df.columns and x_axis in df.columns:
            plot_df = df[[x_axis, metric_y1]].dropna()

            if not plot_df.empty:
                x_data = plot_df[x_axis]
                y_data = apply_moving_average(plot_df[metric_y1], smoothing)

                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=y_data,
                        mode="lines",
                        name=f"{run_name} - {metric_y1}",
                        line=dict(color=color),
                    ),
                    secondary_y=False,
                )

        # Secondary metric
        if has_secondary and metric_y2 in df.columns and x_axis in df.columns:
            plot_df_sec = df[[x_axis, metric_y2]].dropna()

            if not plot_df_sec.empty:
                x_data_sec = plot_df_sec[x_axis]
                y_data_sec = apply_moving_average(plot_df_sec[metric_y2], smoothing)

                fig.add_trace(
                    go.Scatter(
                        x=x_data_sec,
                        y=y_data_sec,
                        mode="lines",
                        name=f"{run_name} - {metric_y2}",
                        line=dict(color=color, dash="dash"),
                    ),
                    secondary_y=True,
                )

    fig.update_xaxes(title_text=x_axis)
    fig.update_yaxes(title_text=metric_y1, secondary_y=False)
    if has_secondary:
        fig.update_yaxes(title_text=metric_y2, secondary_y=True)

    fig.update_layout(
        height=500,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig


def create_simulation_metrics_chart(
    df: pd.DataFrame,
    metrics: List[str],
    x_axis: str = "day",
    show_std: bool = True,
) -> go.Figure:
    """
    Create a chart for simulation metrics over time.

    Args:
        df: DataFrame with day and metric columns.
        metrics: List of metric names to plot.
        x_axis: X-axis column name.
        show_std: Whether to show confidence bands.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set1

    for i, metric in enumerate(metrics):
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"

        if mean_col not in df.columns:
            continue

        color = colors[i % len(colors)]
        x_data = df[x_axis]
        y_mean = df[mean_col]

        # Main line
        fig.add_trace(
            go.Scatter(
                x=x_data,
                y=y_mean,
                mode="lines+markers",
                name=metric,
                line=dict(color=color),
                marker=dict(size=6),
            )
        )

        # Confidence band
        if show_std and std_col in df.columns:
            y_std = df[std_col]
            y_upper = y_mean + y_std
            y_lower = y_mean - y_std

            fig.add_trace(
                go.Scatter(
                    x=pd.concat([x_data, x_data[::-1]]),
                    y=pd.concat([y_upper, y_lower[::-1]]),
                    fill="toself",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ",0.2)"),
                    line=dict(color="rgba(0,0,0,0)"),
                    name=f"{metric} +/- std",
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    fig.update_layout(
        xaxis_title=x_axis,
        yaxis_title="Value",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig


def create_radar_chart(
    policy_metrics: Dict[str, Dict[str, float]],
    metrics: List[str],
) -> go.Figure:
    """
    Create a radar/spider chart comparing metrics across policies.

    Each metric is normalized to 0-1 scale for fair comparison.

    Args:
        policy_metrics: Dict mapping policy name -> {metric: value}.
        metrics: List of metric names to include as radar axes.

    Returns:
        Plotly Figure object.
    """
    if not policy_metrics or not metrics:
        return go.Figure()

    # Compute min/max for normalization
    metric_ranges: Dict[str, tuple] = {}
    for metric in metrics:
        values = [pm.get(metric, 0) for pm in policy_metrics.values()]
        min_val = min(values) if values else 0
        max_val = max(values) if values else 1
        metric_ranges[metric] = (min_val, max_val)

    colors = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, (policy, pm) in enumerate(policy_metrics.items()):
        normalized = []
        for metric in metrics:
            val = pm.get(metric, 0)
            mn, mx = metric_ranges[metric]
            norm = (val - mn) / (mx - mn) if mx > mn else 0.5
            normalized.append(norm)

        # Close the polygon
        normalized.append(normalized[0])
        labels = [m.replace("_", " ").capitalize() for m in metrics]
        labels.append(labels[0])

        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatterpolar(
                r=normalized,
                theta=labels,
                fill="toself",
                name=policy,
                line=dict(color=color),
                opacity=0.85,
            )
        )

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False),
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
        height=420,
        font_family=PLOTLY_LAYOUT_DEFAULTS["font_family"],
        paper_bgcolor=PLOTLY_LAYOUT_DEFAULTS["paper_bgcolor"],
        margin=PLOTLY_LAYOUT_DEFAULTS["margin"],
    )

    return fig


def create_kpi_cards_html(metrics: Dict[str, Any], prefix: str = "") -> str:
    """
    Generate HTML for KPI metric cards.

    Args:
        metrics: Dict of metric_name -> value.
        prefix: Optional prefix for metric display names.

    Returns:
        HTML string for the KPI cards.
    """
    from logic.src.pipeline.ui.styles.styling import (
        KPI_COLORS,
        KPI_FALLBACK_COLORS,
        create_kpi_html,
    )

    cards_html = '<div style="display: flex; flex-wrap: wrap; gap: 14px;">'

    for i, (name, value) in enumerate(metrics.items()):
        display_name = f"{prefix}{name}" if prefix else name

        if isinstance(value, float):
            formatted_value = f"{value:,.2f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        if display_name in KPI_COLORS:
            color, color_end = KPI_COLORS[display_name]
        else:
            fallback = KPI_FALLBACK_COLORS[i % len(KPI_FALLBACK_COLORS)]
            color, color_end = fallback

        cards_html += create_kpi_html(display_name, formatted_value, color, color_end)

    cards_html += "</div>"
    return cards_html


def create_bar_chart(
    data: Dict[str, float],
    title: str = "Comparison",
    x_label: str = "Category",
    y_label: str = "Value",
) -> go.Figure:
    """
    Create a simple bar chart.

    Args:
        data: Dict mapping category names to values.
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(
        data=[
            go.Bar(
                x=list(data.keys()),
                y=list(data.values()),
                marker_color=px.colors.qualitative.Set2,
            )
        ]
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig
