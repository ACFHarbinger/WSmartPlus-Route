# Copyright (c) WSmart-Route. All rights reserved.
"""
Plotly charts for benchmark visualization.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from logic.src.pipeline.ui.components.charts import PLOTLY_LAYOUT_DEFAULTS


def create_benchmark_comparison_chart(
    df: pd.DataFrame, metric: str, title: str, x_axis: str = "policy", color_col: str = "num_nodes"
) -> go.Figure:
    """
    Create a bar chart comparing benchmark metrics across policies/models.
    """
    if df.empty or metric not in df.columns:
        return go.Figure()

    # Determine the best x_axis if default "policy" is missing or all NaN
    possible_x = ["policy", "benchmark", "message", "model"]

    selected_x = x_axis
    if selected_x not in df.columns or df[selected_x].isna().all():
        for col in possible_x:
            if col in df.columns and not df[col].isna().all():
                selected_x = col
                break

    # Final fallback to index if no categorical columns have data
    if selected_x not in df.columns or df[selected_x].isna().all():
        df_copy = df.copy()
        df_copy["index"] = range(len(df_copy))
        selected_x = "index"
    else:
        df_copy = df

    # Drop rows where y-axis metric is NaN to avoid empty bars/errors
    df_plot = df_copy.dropna(subset=[metric])
    if df_plot.empty:
        return go.Figure()

    # Convert numeric color column to string for discrete color mapping if needed
    if color_col in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[color_col]):
        df_plot[color_col] = df_plot[color_col].astype(str)

    fig = px.bar(
        df_plot,
        x=selected_x,
        y=metric,
        color=color_col if color_col in df_plot.columns else None,
        barmode="group",
        title=title,
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    fig.update_layout(
        height=400,
        xaxis_title=selected_x.replace("_", " ").capitalize(),
        yaxis_title=metric.replace("_", " ").capitalize(),
        legend_title=color_col.capitalize() if color_col in df_plot.columns else None,
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig


def create_latency_throughput_scatter(
    df: pd.DataFrame, latency_col: str = "latency", throughput_col: str = "throughput", color_col: str = "num_nodes"
) -> go.Figure:
    """
    Create a scatter plot for latency vs throughput.
    """
    if df.empty or latency_col not in df.columns or throughput_col not in df.columns:
        return go.Figure()

    # Create plot data and drop NaNs in key columns
    df_plot = df.copy()

    # Required columns for coordinates
    cols_to_check = [latency_col, throughput_col]

    # Add optional columns for encoding if they exist
    size_col = "batch_size" if "batch_size" in df.columns else None
    if size_col:
        cols_to_check.append(size_col)

    # Drop rows with NaNs in required/encoding columns
    df_plot = df_plot.dropna(subset=[c for c in cols_to_check if c in df_plot.columns])

    if df_plot.empty:
        return go.Figure()

    # Convert numeric color column to string
    if color_col in df_plot.columns and pd.api.types.is_numeric_dtype(df_plot[color_col]):
        df_plot[color_col] = df_plot[color_col].astype(str)

    fig = px.scatter(
        df_plot,
        x=latency_col,
        y=throughput_col,
        color=color_col if color_col in df_plot.columns else None,
        size=size_col,
        hover_data=["num_nodes", "batch_size"]
        if "num_nodes" in df_plot.columns and "batch_size" in df_plot.columns
        else None,
        title="Latency vs Throughput",
        color_discrete_sequence=px.colors.qualitative.Vivid,
    )

    fig.update_layout(
        height=400,
        xaxis_title="Latency (s)",
        yaxis_title="Throughput (inst/s)",
        **PLOTLY_LAYOUT_DEFAULTS,
    )

    return fig
