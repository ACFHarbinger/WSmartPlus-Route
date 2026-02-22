"""
Chart rendering functions for the Data Explorer page.

Extracted from ``data_explorer.py`` to keep module sizes under 400 LoC.
"""

from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.src.ui.components.charts import (
    PLOTLY_LAYOUT_DEFAULTS,
    calculate_pareto_front,
    create_area_chart,
    create_box_plot_chart,
    create_correlation_matrix_chart,
    create_heatmap_chart,
    create_histogram_chart,
    create_multi_y_line_chart,
    create_pareto_scatter_chart,
)

_CHART_TYPES = [
    "Line Chart",
    "Bar Chart",
    "Scatter Plot",
    "Area Chart",
    "Histogram",
    "Box Plot",
    "Heatmap",
    "Correlation Matrix",
]


def _resolve_column(columns: List[Any], col_text: str) -> Any:
    """Resolve a column name, handling integer column names from combo text."""
    if col_text in columns:
        return col_text
    try:
        int_key = int(col_text)
        if int_key in columns:
            return int_key
    except ValueError:
        pass
    return None


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    """Return string names of numeric columns."""
    return [str(c) for c in df.select_dtypes(include=["number"]).columns]


def _has_distribution_meta(df: pd.DataFrame) -> bool:
    """Check whether DataFrame contains simulation distribution metadata."""
    return "__Distributions__" in df.columns


def _unique_distributions(df: pd.DataFrame) -> List[str]:
    """Return sorted unique distribution values."""
    if "__Distributions__" not in df.columns:
        return []
    return sorted(df["__Distributions__"].dropna().unique().tolist())


def _render_visualization_tab(df: pd.DataFrame) -> None:
    """Render the enhanced Visualization tab."""

    str_columns = [str(c) for c in df.columns.tolist()]
    numeric_cols = _numeric_columns(df)
    has_dist = _has_distribution_meta(df)

    # --- Controls row ---
    ctrl_cols = st.columns([2, 2, 2, 2] if has_dist else [2, 2, 2])

    with ctrl_cols[0]:
        chart_type = st.selectbox("Chart Type", options=_CHART_TYPES, index=0, key="de_chart_type")

    # Distribution filter (only for simulation JSON data)
    dist_filter = "All"
    if has_dist:
        with ctrl_cols[-1]:
            dist_options = ["All"] + _unique_distributions(df)
            dist_filter = st.selectbox("Distribution", options=dist_options, index=0, key="de_dist_filter")

    # Charts that don't need X/Y selection
    no_xy_charts = {"Heatmap", "Correlation Matrix"}
    # Charts that only need Y (or column selection)
    y_only_charts = {"Histogram", "Box Plot"}

    needs_x = chart_type not in no_xy_charts and chart_type not in y_only_charts
    needs_y = chart_type not in no_xy_charts

    with ctrl_cols[1]:
        if needs_x:
            x_col = st.selectbox("X Axis", options=str_columns, index=0, key="de_x_axis")
        else:
            x_col = ""
            st.selectbox("X Axis", options=str_columns, index=0, disabled=True, key="de_x_axis_disabled")

    if needs_y:
        with ctrl_cols[2] if len(ctrl_cols) > 2 else ctrl_cols[1]:
            if chart_type == "Box Plot":
                y_cols = st.multiselect(
                    "Columns",
                    options=numeric_cols,
                    default=numeric_cols[:3] if len(numeric_cols) >= 3 else numeric_cols,
                    key="de_box_cols",
                )
            elif chart_type == "Histogram":
                y_col = st.selectbox(
                    "Column",
                    options=numeric_cols if numeric_cols else str_columns,
                    index=0,
                    key="de_hist_col",
                )
            elif chart_type in ("Line Chart", "Bar Chart"):
                # Multi-Y support for line/bar
                y_cols = st.multiselect(
                    "Y Axis (multi)",
                    options=str_columns,
                    default=[str_columns[min(1, len(str_columns) - 1)]],
                    key="de_y_multi",
                )
            else:
                y_idx = min(1, len(str_columns) - 1)
                y_col = st.selectbox("Y Axis", options=str_columns, index=y_idx, key="de_y_axis")

    # --- Advanced options expander ---
    extra_opts: Dict[str, Any] = {}
    if chart_type in ("Scatter Plot", "Histogram"):
        with st.expander("Advanced Options", expanded=False):
            if chart_type == "Scatter Plot":
                cat_columns = ["None"] + [str(c) for c in df.select_dtypes(include=["object", "category"]).columns]
                if has_dist:
                    cat_columns = ["None"] + ["__Distributions__", "__Policy_Names__"] + cat_columns[1:]
                extra_opts["color_by"] = st.selectbox("Color by", options=cat_columns, index=0, key="de_color_by")
                extra_opts["pareto"] = st.checkbox("Show Pareto Front (min X, max Y)", value=False, key="de_pareto")
            elif chart_type == "Histogram":
                extra_opts["nbins"] = st.slider("Number of bins", 5, 100, 30, key="de_nbins")

    # --- Apply distribution filter ---
    plot_df = df
    if has_dist and dist_filter != "All":
        mask = df["__Distributions__"] == dist_filter
        plot_df = pd.DataFrame(df.loc[mask]).reset_index(drop=True)
        if plot_df.empty:
            st.warning(f"No data for distribution: {dist_filter}")
            return

    # --- Render chart ---
    _render_selected_chart(plot_df, chart_type, x_col, extra_opts, locals())


def _render_line_bar_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    y_cols: List[str],
) -> None:
    """Render Line Chart or Bar Chart with multi-Y support."""
    columns = df.columns.tolist()

    if not y_cols:
        st.warning("Select at least one Y column.")
        return

    if len(y_cols) > 1 and chart_type == "Line Chart":
        x_key = _resolve_column(columns, x_col)
        if x_key is not None:
            fig = create_multi_y_line_chart(df, x_key, y_cols, title=f"Multi-Series: {', '.join(y_cols)}")
            st.plotly_chart(fig, width="stretch")
        else:
            st.error(f"Column not found: {x_col}")
        return

    y_col_str = y_cols[0]
    x_key = _resolve_column(columns, x_col)
    y_key = _resolve_column(columns, y_col_str)
    if x_key is None or y_key is None:
        st.error(f"Column not found: {x_col if x_key is None else y_col_str}")
        return

    x_data = df[x_key]
    y_data = df[y_key]

    if chart_type == "Line Chart":
        sorted_tmp = pd.DataFrame({"x": x_data, "y": y_data}).sort_values("x")
        fig = go.Figure(
            go.Scatter(
                x=sorted_tmp["x"],
                y=sorted_tmp["y"],
                mode="lines+markers",
                hovertemplate=f"{y_col_str}: %{{y:.4f}}<extra>%{{x}}</extra>",
            )
        )
        fig.update_layout(
            title=f"Line: {y_col_str} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col_str,
            height=400,
            **PLOTLY_LAYOUT_DEFAULTS,
        )
    elif len(y_cols) > 1:
        fig = go.Figure()
        colors = px.colors.qualitative.Set2
        for i, yc in enumerate(y_cols):
            yk = _resolve_column(columns, yc)
            if yk is not None:
                fig.add_trace(go.Bar(x=x_data, y=df[yk], name=str(yc), marker_color=colors[i % len(colors)]))
        fig.update_layout(
            barmode="group",
            title=f"Bar: {', '.join(y_cols)} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title="Value",
            height=400,
            **PLOTLY_LAYOUT_DEFAULTS,
        )
    else:
        fig = go.Figure(
            go.Bar(
                x=x_data,
                y=y_data,
                marker_color=px.colors.qualitative.Set2,
                hovertemplate=f"{y_col_str}: %{{y:.4f}}<extra>%{{x}}</extra>",
            )
        )
        fig.update_layout(
            title=f"Bar: {y_col_str} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col_str,
            height=400,
            **PLOTLY_LAYOUT_DEFAULTS,
        )

    st.plotly_chart(fig, width="stretch")


def _render_scatter_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    extra_opts: Dict[str, Any],
) -> None:
    """Render Scatter Plot with optional color-by and Pareto front."""
    columns = df.columns.tolist()
    x_key = _resolve_column(columns, x_col)
    y_key = _resolve_column(columns, y_col)
    if x_key is None or y_key is None:
        st.error(f"Column not found: {x_col if x_key is None else y_col}")
        return

    x_data = df[x_key]
    y_data = df[y_key]
    color_col = extra_opts.get("color_by", "None")
    pareto = extra_opts.get("pareto", False)

    color_series = None
    if color_col != "None" and color_col in df.columns:
        color_series = df[color_col]

    pareto_indices = None
    if pareto:
        try:
            pareto_indices = calculate_pareto_front(x_data.tolist(), y_data.tolist())
        except Exception:
            pareto_indices = None

    fig = create_pareto_scatter_chart(
        x_data,
        y_data,
        x_label=x_col,
        y_label=y_col,
        pareto_indices=pareto_indices,
        color_by=color_series,
        title=f"Scatter: {y_col} vs {x_col}",
    )
    st.plotly_chart(fig, width="stretch")


def _render_area_chart(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
) -> None:
    """Render an Area Chart."""
    columns = df.columns.tolist()
    x_key = _resolve_column(columns, x_col)
    y_key = _resolve_column(columns, y_col)
    if x_key is None or y_key is None:
        st.error(f"Column not found: {x_col if x_key is None else y_col}")
        return

    sorted_tmp = pd.DataFrame({"x": df[x_key], "y": df[y_key]}).sort_values("x")
    fig = create_area_chart(
        sorted_tmp["x"],
        sorted_tmp["y"],
        x_label=x_col,
        y_label=y_col,
        title=f"Area: {y_col} vs {x_col}",
    )
    st.plotly_chart(fig, width="stretch")


def _render_selected_chart(
    df: pd.DataFrame,
    chart_type: str,
    x_col: str,
    extra_opts: Dict[str, Any],
    local_vars: Dict[str, Any],
) -> None:
    """Dispatch to the appropriate chart renderer."""
    columns = df.columns.tolist()

    if chart_type == "Heatmap":
        st.plotly_chart(create_heatmap_chart(df, title="Heatmap"), width="stretch")
    elif chart_type == "Correlation Matrix":
        st.plotly_chart(create_correlation_matrix_chart(df), width="stretch")
    elif chart_type == "Histogram":
        y_col = local_vars.get("y_col", "")
        col_key = _resolve_column(columns, y_col)
        if col_key is not None:
            nbins = extra_opts.get("nbins", 30)
            st.plotly_chart(
                create_histogram_chart(df[col_key], nbins=nbins, title=f"Histogram: {y_col}"),
                width="stretch",
            )
        else:
            st.warning("Select a valid column.")
    elif chart_type == "Box Plot":
        y_cols = local_vars.get("y_cols", [])
        if y_cols:
            resolved = [str(c) for c in columns if str(c) in y_cols]
            st.plotly_chart(
                create_box_plot_chart(df, resolved, title="Box Plot: Distribution Comparison"),
                width="stretch",
            )
        else:
            st.warning("Select at least one column.")
    elif chart_type in ("Line Chart", "Bar Chart"):
        _render_line_bar_chart(df, chart_type, x_col, local_vars.get("y_cols", []))
    elif chart_type == "Scatter Plot":
        _render_scatter_chart(df, x_col, local_vars.get("y_col", ""), extra_opts)
    elif chart_type == "Area Chart":
        _render_area_chart(df, x_col, local_vars.get("y_col", ""))
    else:
        st.warning(f"Unknown chart type: {chart_type}")
