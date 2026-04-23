"""Plotly chart generators for the dashboard.

This module provides reusable visualization components for cross-mode
performance auditing. It supports sparklines, multi-axis training curves,
radar/spider policy comparisons, and geospatial heatmaps.

Example:
    spark_html = create_sparkline_svg([0.1, 0.4, 0.2, 0.7])

Attributes:
    apply_moving_average: Utility to smooth noisy telemetry data.
    create_sparkline_svg: Lightweight SVG generator for inline KPIs.
    create_training_loss_chart: Multi-run comparison with dual Y-axes.
"""

from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from logic.src.constants.dashboard import PLOTLY_LAYOUT_DEFAULTS


def apply_moving_average(data: pd.Series, window: int) -> pd.Series:
    """Applies moving average smoothing to a numeric series.

    Args:
        data: Input telemetry series.
        window: Sliding window size.

    Returns:
        pd.Series: Smoothed data series.
    """
    if window <= 1:
        return data
    return data.rolling(window=window, min_periods=1).mean()


def create_sparkline_svg(
    values: List[float],
    width: int = 60,
    height: int = 20,
    color: str = "rgba(255,255,255,0.8)",
) -> str:
    """Creates a lightweight inline SVG sparkline for KPI cards.

    Args:
        values: Data points for the sparkline.
        width: SVG width in pixels.
        height: SVG height in pixels.
        color: Stroke color for the line.

    Returns:
        SVG string suitable for embedding in HTML.
    """
    if not values or len(values) < 2:
        return ""

    min_val = min(values)
    max_val = max(values)
    val_range = max_val - min_val if max_val > min_val else 1.0

    padding = 2
    plot_w = width - 2 * padding
    plot_h = height - 2 * padding

    points = []
    for i, v in enumerate(values):
        x = padding + (i / (len(values) - 1)) * plot_w
        y = padding + plot_h - ((v - min_val) / val_range) * plot_h
        points.append(f"{x:.1f},{y:.1f}")

    polyline_points = " ".join(points)

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<polyline points="{polyline_points}" fill="none" '
        f'stroke="{color}" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>'
        f"</svg>"
    )


def create_training_loss_chart(
    runs_data: Dict[str, pd.DataFrame],
    metric_y1: str = "train_loss",
    metric_y2: Optional[str] = "val_cost",
    x_axis: str = "epoch",
    smoothing: int = 1,
) -> go.Figure:
    """Creates a multi-run comparison chart with optional dual Y-axes.

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
                        hovertemplate=f"{metric_y1}: %{{y:.4f}}<extra>{run_name}</extra>",
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
                        hovertemplate=f"{metric_y2}: %{{y:.4f}}<extra>{run_name}</extra>",
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
    """Creates a line chart for simulation metrics over time with bands.

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
                hovertemplate=f"{metric}: %{{y:.2f}}<extra>Day %{{x}}</extra>",
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
    """Creates a radar/spider chart comparing policies across N dimensions.

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


def create_stacked_bar_chart(
    categories: List[str],
    series: Dict[str, List[float]],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
    colors: Optional[List[str]] = None,
) -> go.Figure:
    """Creates a stacked bar chart for categorical multi-series data.

    Args:
        categories: X-axis category labels.
        series: Dict mapping series name -> list of values (same length as categories).
        title: Chart title.
        x_label: X-axis label.
        y_label: Y-axis label.
        colors: Optional list of colors for each series.

    Returns:
        Plotly Figure object.
    """
    default_colors = px.colors.qualitative.Set2
    fig = go.Figure()

    for i, (name, values) in enumerate(series.items()):
        bar_color = colors[i] if colors and i < len(colors) else default_colors[i % len(default_colors)]
        fig.add_trace(
            go.Bar(
                x=categories,
                y=values,
                name=name,
                marker_color=bar_color,
                hovertemplate=f"{name}: %{{y:.2f}}<extra>%{{x}}</extra>",
            )
        )

    fig.update_layout(
        barmode="stack",
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig


def create_heatmap_chart(
    df: pd.DataFrame,
    title: str = "Heatmap",
) -> go.Figure:
    """Creates an interactive Plotly heatmap from a numeric DataFrame.

    Args:
        df: DataFrame with numeric values. Rows are Y-axis, columns are X-axis.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No numeric data for heatmap", showarrow=False)
        return fig

    fig = go.Figure(
        data=go.Heatmap(
            z=numeric_df.values,
            x=[str(c) for c in numeric_df.columns],
            y=list(range(len(numeric_df))),
            colorscale="Viridis",
            hovertemplate="Row %{y}, Col %{x}<br>Value: %{z:.4f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Columns",
        yaxis_title="Row Index",
        height=500,
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig
