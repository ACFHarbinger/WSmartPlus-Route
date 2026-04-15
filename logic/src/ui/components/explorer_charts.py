# Copyright (c) WSmart-Route. All rights reserved.
"""
Plotly chart generators for the Data Explorer and analysis pages.

Extracted from ``charts.py`` to keep module sizes under 400 LoC.
Functions here are re-exported from ``charts.py`` for backward compatibility.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from logic.src.constants.dashboard import PLOTLY_LAYOUT_DEFAULTS


def create_area_chart(
    x: Any,
    y: Any,
    x_label: str = "X",
    y_label: str = "Y",
    title: str = "Area Chart",
) -> go.Figure:
    """
    Create a filled area chart with a line overlay.

    Args:
        x: X-axis data (array-like).
        y: Y-axis data (array-like).
        x_label: X-axis label.
        y_label: Y-axis label.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines",
            fill="tozeroy",
            fillcolor="rgba(31, 119, 180, 0.3)",
            line=dict(color="#1f77b4"),
            name=y_label,
            hovertemplate=f"{y_label}: %{{y:.4f}}<extra>%{{x}}</extra>",
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=400,
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig


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
                hovertemplate="%{x}<br>%{y:.2f}<extra></extra>",
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


def calculate_pareto_front(x_values: List[float], y_values: List[float]) -> List[int]:
    """
    Calculate indices of non-dominated points (minimise X, maximise Y).

    Ported from ``gui/src/tabs/analysis/output_analysis/engine.py``.

    Args:
        x_values: X coordinates.
        y_values: Y coordinates.

    Returns:
        Indices of points on the Pareto front.
    """
    n = len(x_values)
    pareto_indices: List[int] = []
    for i in range(n):
        dominated = False
        for j in range(n):
            if i == j:
                continue
            if (
                x_values[j] <= x_values[i]
                and y_values[j] >= y_values[i]
                and (x_values[j] < x_values[i] or y_values[j] > y_values[i])
            ):
                dominated = True
                break
        if not dominated:
            pareto_indices.append(i)
    return pareto_indices


def create_histogram_chart(
    series: pd.Series,
    nbins: int = 30,
    title: str = "Histogram",
) -> go.Figure:
    """
    Create a Plotly histogram.

    Args:
        series: Data series to plot.
        nbins: Number of bins.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure(
        data=go.Histogram(
            x=series,
            nbinsx=nbins,
            marker_color="#1f77b4",
            opacity=0.85,
            hovertemplate="Range: %{x}<br>Count: %{y}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        xaxis_title=series.name if series.name else "Value",
        yaxis_title="Count",
        height=400,
        bargap=0.05,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def create_box_plot_chart(
    df: pd.DataFrame,
    columns: List[str],
    title: str = "Box Plot",
) -> go.Figure:
    """
    Create a multi-column box plot for distribution comparison.

    Args:
        df: Source DataFrame.
        columns: Column names to include.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, col in enumerate(columns):
        if col in df.columns:
            fig.add_trace(
                go.Box(
                    y=df[col],
                    name=str(col),
                    marker_color=colors[i % len(colors)],
                    boxmean="sd",
                )
            )
    fig.update_layout(
        title=title,
        yaxis_title="Value",
        height=400,
        showlegend=False,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def create_correlation_matrix_chart(
    df: pd.DataFrame,
    title: str = "Correlation Matrix",
) -> go.Figure:
    """
    Create a correlation heatmap from numeric DataFrame columns.

    Args:
        df: Source DataFrame (numeric columns used).
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    numeric_df = df.select_dtypes(include=["number"])
    if numeric_df.shape[1] < 2:
        fig = go.Figure()
        fig.add_annotation(text="Need at least 2 numeric columns", showarrow=False)
        return fig

    corr = numeric_df.corr()

    fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=[str(c) for c in corr.columns],
            y=[str(c) for c in corr.columns],
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            hovertemplate="%{x} vs %{y}<br>r = %{z:.3f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        height=500,
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def create_pareto_scatter_chart(
    x: Any,
    y: Any,
    x_label: str = "X",
    y_label: str = "Y",
    pareto_indices: Optional[List[int]] = None,
    color_by: Optional[pd.Series] = None,
    title: str = "Scatter with Pareto Front",
) -> go.Figure:
    """
    Create a scatter plot with optional Pareto front overlay.

    Args:
        x: X-axis data.
        y: Y-axis data.
        x_label: X-axis label.
        y_label: Y-axis label.
        pareto_indices: Indices of Pareto-optimal points (drawn as connected line).
        color_by: Optional categorical series for colour grouping.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()

    if color_by is not None:
        for group_name in color_by.unique():
            mask = color_by == group_name
            fig.add_trace(
                go.Scatter(
                    x=np.asarray(x)[mask],
                    y=np.asarray(y)[mask],
                    mode="markers",
                    name=str(group_name),
                    marker=dict(size=8, opacity=0.7),
                    hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.4f}}<extra>{group_name}</extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name="Data",
                marker=dict(size=8, opacity=0.7, color="#1f77b4"),
                hovertemplate=f"{x_label}: %{{x}}<br>{y_label}: %{{y:.4f}}<extra></extra>",
            )
        )

    if pareto_indices:
        x_arr = np.asarray(x)
        y_arr = np.asarray(y)
        pts = sorted(
            [(x_arr[i], y_arr[i]) for i in pareto_indices],
            key=lambda p: p[0],
        )
        px_vals, py_vals = zip(*pts, strict=False)
        fig.add_trace(
            go.Scatter(
                x=px_vals,
                y=py_vals,
                mode="lines+markers",
                name="Pareto Front",
                line=dict(color="#e8eaed", width=2, dash="dash"),
                marker=dict(size=10, symbol="star", color="gold", line=dict(color="#e8eaed", width=1)),
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=x_label,
        yaxis_title=y_label,
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig


def create_multi_y_line_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    title: str = "Multi-Series Line Chart",
) -> go.Figure:
    """
    Create a line chart with multiple Y-series overlaid.

    Args:
        df: Source DataFrame.
        x_col: X-axis column name.
        y_cols: List of Y-axis column names.
        title: Chart title.

    Returns:
        Plotly Figure object.
    """
    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    sorted_df = df.sort_values(x_col)

    for i, col in enumerate(y_cols):
        if col not in sorted_df.columns:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=sorted_df[x_col],
                y=sorted_df[col],
                mode="lines+markers",
                name=str(col),
                line=dict(color=color),
                marker=dict(size=5),
                hovertemplate=f"{col}: %{{y:.4f}}<extra>%{{x}}</extra>",
            )
        )

    fig.update_layout(
        title=title,
        xaxis_title=str(x_col),
        yaxis_title="Value",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        **PLOTLY_LAYOUT_DEFAULTS,
    )
    return fig
