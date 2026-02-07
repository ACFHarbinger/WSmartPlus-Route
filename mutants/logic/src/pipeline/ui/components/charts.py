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
            # Filter rows where both x and y are not NaN to ensure alignment
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
            # Filter rows where both x and y are not NaN to ensure alignment
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
                    name=f"{metric} ± std",
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
    cards_html = '<div style="display: flex; flex-wrap: wrap; gap: 16px;">'

    for name, value in metrics.items():
        display_name = f"{prefix}{name}" if prefix else name

        if isinstance(value, float):
            formatted_value = f"{value:,.2f}"
        elif isinstance(value, int):
            formatted_value = f"{value:,}"
        else:
            formatted_value = str(value)

        card = (
            f'<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); '
            f"border-radius: 12px; padding: 20px; min-width: 150px; text-align: center; "
            f'color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">'
            f'<div style="font-size: 14px; opacity: 0.9; margin-bottom: 8px;">{display_name}</div>'
            f'<div style="font-size: 24px; font-weight: bold;">{formatted_value}</div>'
            f"</div>"
        )
        cards_html += card

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
    )

    return fig
