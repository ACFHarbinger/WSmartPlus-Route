"""Chart rendering functions for the Data Explorer page.

This module provides specialized visualization components for the interactive
data exploration dashboard. It includes multi-series line charts, Pareto
tradeoff scatter plots, TensorDict coordinate maps, and statistical
distribution analyzers for DRL datasets.

Example:
    import pandas as pd
    df = pd.DataFrame({"x": [1, 2], "y": [10, 20]})
    render_summary_stats(df)

Attributes:
    _CHART_TYPES: Supported visualization modes for the explorer.
    _numeric_columns: Utility to discover numerical data in DataFrames.
    _resolve_column: Maps UI column labels to internal DataFrame keys.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.src.constants.dashboard import PLOTLY_LAYOUT_DEFAULTS
from logic.src.ui.components.charts import create_heatmap_chart
from logic.src.ui.components.explorer_charts import (
    calculate_pareto_front,
    create_area_chart,
    create_box_plot_chart,
    create_correlation_matrix_chart,
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
    """Resolves a UI column name string to the actual DataFrame column key.

    Handles mixed-type column keys (e.g., integers from NumPy arrays).

    Args:
        columns: List of valid column keys from the target DataFrame.
        col_text: The user-selected column label from a UI component.

    Returns:
        Any: The resolved column key or None if not found.
    """
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
    """Retrieves string labels for all numerical columns in a DataFrame.

    Args:
        df: Input DataFrame to scan.

    Returns:
        List[str]: String representations of numerical column keys.
    """
    return [str(c) for c in df.select_dtypes(include=["number"]).columns]


def _has_distribution_meta(df: pd.DataFrame) -> bool:
    """Checks if a DataFrame contains simulation distribution metadata.

    Args:
        df: Input DataFrame to scan.

    Returns:
        bool: True if distribution metadata columns are present.
    """
    return "__Distributions__" in df.columns


def _unique_distributions(df: pd.DataFrame) -> List[str]:
    """Retrieves a sorted list of unique simulation distribution labels.

    Args:
        df: Input DataFrame containing '__Distributions__' metadata.

    Returns:
        List[str]: Sorted unique distribution names.
    """
    if "__Distributions__" not in df.columns:
        return []
    return sorted(df["__Distributions__"].dropna().unique().tolist())


def _render_visualization_tab(df: pd.DataFrame) -> None:
    """Renders the comprehensive Visualization tab for the Data Explorer.

    Args:
        df: The currently active DataFrame to visualize.
    """

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
    """Renders Line or Bar charts with multi-series support.

    Args:
        df: The dataset containing plotting data.
        chart_type: Either 'Line Chart' or 'Bar Chart'.
        x_col: The column label for the horizontal axis.
        y_cols: List of column labels for vertical axes/series.
    """
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
    """Renders Scatter plots with optional Pareto and color mapping.

    Args:
        df: The dataset containing plotting data.
        x_col: The column label for the horizontal axis.
        y_col: The column label for the vertical axis.
        extra_opts: Visual configuration including 'color_by' and 'pareto'.
    """
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
    """Renders an Area Chart for temporal or sequential data.

    Args:
        df: The dataset containing plotting data.
        x_col: The column label for the horizontal axis.
        y_col: The column label for the vertical axis.
    """
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
    """Dispatches plot requests to specialized visual renderers.

    Args:
        df: The dataset containing plotting data.
        chart_type: The target visualization mode from _CHART_TYPES.
        x_col: The primary horizontal axis column label.
        extra_opts: Rendering configuration for scatter/histograms.
        local_vars: Context-specific variables for multi-series charts.
    """
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


# ---------------------------------------------------------------------------
# TensorDict overview tab
# ---------------------------------------------------------------------------


def _find_td_table(tables: Dict[str, pd.DataFrame], key: str) -> Optional[pd.DataFrame]:
    """Retrieves a specific tensor-derived table by key identifier.

    Handles the label formatting used in TensorDict discovery.

    Args:
        tables: Mapping of table labels to DataFrames.
        key: The raw tensor key to find.

    Returns:
        Optional[pd.DataFrame]: The matched DataFrame or None.
    """
    for label, df in tables.items():
        if label.startswith(f"{key} ("):
            return df
    return None


def _render_td_coord_section(
    tables: Dict[str, pd.DataFrame],
    coord_keys: List[str],
    depot_keys: List[str],
    lazy_loader: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
) -> None:
    """Renders interactive coordinate scatter maps with optional depot overlay.

    Args:
        tables: Mapping of already loaded DataFrames.
        coord_keys: Keys identified as geospatial coordinate tensors.
        depot_keys: Keys identified as single-point depot tensors.
        lazy_loader: Optional callable for on-demand loading of large tensors.
    """
    st.subheader("Coordinate Map")

    ctrl = st.columns([2, 2, 2])
    with ctrl[0]:
        coord_key = st.selectbox("Coordinate Key", options=coord_keys, key="td_coord_key")

    coord_df = _find_td_table(tables, coord_key)
    if coord_df is None and lazy_loader is not None:
        with st.spinner(f"Loading '{coord_key}'…"):
            coord_df = lazy_loader(coord_key)
    if coord_df is None or "x" not in coord_df.columns:
        st.warning(f"Could not find coordinate data for key '{coord_key}'.")
        return

    max_sample = int(coord_df["sample_id"].max()) if "sample_id" in coord_df.columns else 0

    with ctrl[1]:
        sample_mode = st.selectbox(
            "Display Mode",
            options=["Single Sample", "All Samples (up to 30)"],
            key="td_sample_mode",
        )

    if sample_mode == "Single Sample":
        with ctrl[2]:
            sample_idx = st.number_input(
                "Sample Index",
                min_value=0,
                max_value=max_sample,
                value=0,
                step=1,
                key="td_sample_idx",
            )
        plot_df = coord_df[coord_df["sample_id"] == int(sample_idx)].copy()
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            title=f"Sample {sample_idx}: {coord_key}",
            labels={"x": "X", "y": "Y"},
            opacity=0.85,
        )
        fig.update_traces(marker=dict(size=8, color="#1f77b4"))

        # Overlay depot as a star marker if a depot key exists
        for dk in depot_keys:
            depot_df = _find_td_table(tables, dk)
            if depot_df is None and lazy_loader is not None:
                depot_df = lazy_loader(dk)
            if depot_df is not None and "x" in depot_df.columns and int(sample_idx) < len(depot_df):
                row = depot_df.iloc[int(sample_idx)]
                fig.add_trace(
                    go.Scatter(
                        x=[float(row["x"])],
                        y=[float(row["y"])],
                        mode="markers",
                        marker=dict(size=16, color="#d62728", symbol="star"),
                        name=f"{dk} (depot)",
                    )
                )
    else:
        cap = min(30, max_sample + 1)
        plot_df = coord_df[coord_df["sample_id"] < cap].copy()
        plot_df["sample_id"] = plot_df["sample_id"].astype(str)
        fig = px.scatter(
            plot_df,
            x="x",
            y="y",
            color="sample_id",
            title=f"Coordinate Map: {coord_key} (first {cap} samples)",
            labels={"x": "X", "y": "Y", "sample_id": "Sample"},
            opacity=0.55,
        )

    fig.update_layout(height=480, **PLOTLY_LAYOUT_DEFAULTS)
    st.plotly_chart(fig, use_container_width=True)


def _render_td_dist_section(
    tables: Dict[str, pd.DataFrame],
    scalar_keys: List[str],
    lazy_loader: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
) -> None:
    """Renders statistical distribution views for numerical tensors.

    Calculates histograms and per-node box plots to analyze dataset balance.

    Args:
        tables: Mapping of already loaded DataFrames.
        scalar_keys: Keys identified as numerical/scalar tensors.
        lazy_loader: Optional callable for on-demand loading of large tensors.
    """
    st.subheader("Tensor Distributions")

    sel_key = st.selectbox("Select Key", options=scalar_keys, key="td_dist_key")
    dist_df = _find_td_table(tables, sel_key)
    if dist_df is None and lazy_loader is not None:
        with st.spinner(f"Loading '{sel_key}'…"):
            dist_df = lazy_loader(sel_key)
    if dist_df is None:
        st.warning(f"Could not find data for key '{sel_key}'.")
        return

    numeric_cols = _numeric_columns(dist_df)
    if not numeric_cols:
        st.info("No numeric columns found for this key.")
        return

    nbins = st.slider("Histogram bins", 10, 100, 40, key="td_dist_nbins")

    flat_vals = dist_df[numeric_cols].to_numpy(dtype=float, na_value=np.nan).flatten()
    flat_vals = flat_vals[~np.isnan(flat_vals)]

    col_left, col_right = st.columns(2)

    with col_left:
        st.plotly_chart(
            create_histogram_chart(
                pd.Series(flat_vals),
                nbins=nbins,
                title=f"Value distribution: {sel_key}",
            ),
            use_container_width=True,
        )

    with col_right:
        # Box plot — cap number of columns to keep the chart readable
        MAX_BOX_COLS = 60
        box_cols = numeric_cols[:MAX_BOX_COLS]
        if len(numeric_cols) > MAX_BOX_COLS:
            st.caption(f"Showing first {MAX_BOX_COLS} of {len(numeric_cols)} nodes.")
        st.plotly_chart(
            create_box_plot_chart(dist_df, box_cols, title=f"Per-node spread: {sel_key}"),
            use_container_width=True,
        )


def _render_td_overview_tab(
    td_meta: Dict[str, Any],
    tables: Dict[str, pd.DataFrame],
    lazy_loader: Optional[Callable[[str], Optional[pd.DataFrame]]] = None,
) -> None:
    """Renders the comprehensive structural overview tab for TensorDict files.

    Args:
        td_meta: Metadata dictionary describing tensor shapes and types.
        tables: Mapping of already loaded DataFrames.
        lazy_loader: Optional callable for on-demand loading of large tensors.
    """
    filename = td_meta.get("filename", "unknown")
    batch_size = td_meta.get("batch_size")
    coord_keys: List[str] = td_meta.get("coord_keys", [])
    depot_keys: List[str] = td_meta.get("depot_keys", [])
    scalar_keys: List[str] = td_meta.get("scalar_keys", [])
    summary_rows: List[Dict[str, Any]] = td_meta.get("summary_rows", [])

    # --- Header metrics ---
    m1, m2, m3 = st.columns(3)
    m1.metric("File", filename)
    m2.metric("Instances (batch)", str(batch_size) if batch_size is not None else "—")
    m3.metric("Tensor Keys", str(len(summary_rows)))

    # --- Structure table ---
    if summary_rows:
        st.subheader("Structure")
        summary_df = pd.DataFrame(summary_rows)
        st.dataframe(
            summary_df.style.format(
                {c: "{:.4f}" for c in ["Min", "Max", "Mean", "Std"]},
                na_rep="—",
            ),
            use_container_width=True,
            hide_index=True,
        )

    # --- Coordinate scatter ---
    if coord_keys:
        st.divider()
        _render_td_coord_section(tables, coord_keys, depot_keys, lazy_loader=lazy_loader)

    # --- Distribution analysis ---
    if scalar_keys:
        st.divider()
        _render_td_dist_section(tables, scalar_keys, lazy_loader=lazy_loader)

    if not coord_keys and not scalar_keys:
        st.info("No coordinate or scalar tensors detected for visualisation.")
