"""
Data Explorer mode for the Streamlit dashboard.

Provides interactive file loading, table viewing, and charting for arbitrary
CSV, XLSX, and PKL data files. Includes VRPP/WCVRP dataset splitting heuristics.
"""

from collections import abc
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.src.pipeline.ui.components.charts import (
    PLOTLY_LAYOUT_DEFAULTS,
    create_area_chart,
    create_heatmap_chart,
)

_CHART_TYPES = ["Line Chart", "Bar Chart", "Scatter Plot", "Area Chart", "Heatmap"]


# ---------------------------------------------------------------------------
# Data loading (ported from gui/src/helpers/data_loader_worker.py)
# ---------------------------------------------------------------------------


def _process_raw_to_dfs(raw_data: Any) -> List[pd.DataFrame]:
    """Recursively convert raw data into a flat list of DataFrames."""
    dfs: List[pd.DataFrame] = []

    if isinstance(raw_data, pd.DataFrame):
        dfs.append(raw_data)

    elif isinstance(raw_data, (list, abc.Sequence)) and not isinstance(raw_data, (str, bytes, np.ndarray)):
        try:
            df = pd.DataFrame(list(raw_data))
            if not df.empty:
                return [df]
        except Exception:
            pass

        try:
            array_data = np.array(raw_data)
        except Exception:
            array_data = np.array(raw_data, dtype=object)

        if array_data.dtype == object and array_data.ndim == 1:
            for item in raw_data:
                dfs.extend(_process_raw_to_dfs(item))
        else:
            dfs.extend(_process_raw_to_dfs(array_data))

    elif isinstance(raw_data, np.ndarray):
        if raw_data.ndim >= 3:
            for i in range(raw_data.shape[0]):
                slice_data = raw_data[i, ...].squeeze()
                if slice_data.ndim == 2:
                    dfs.append(pd.DataFrame(slice_data).T)
        elif raw_data.ndim == 2:
            dfs.append(pd.DataFrame(raw_data).T)

    return dfs


def _try_vrpp_split(df: pd.DataFrame) -> Optional[List[Tuple[str, pd.DataFrame]]]:
    """Attempt VRPP/WCVRP 4-column heuristic split."""
    if df.shape[1] != 4:
        return None

    try:
        result: List[Tuple[str, pd.DataFrame]] = []

        # Column 0: Depots
        depot_df = pd.DataFrame(df[0].tolist(), index=df.index)
        result.append(("Depots", depot_df))

        # Column 1: Locations
        loc_df = pd.DataFrame(df[1].tolist(), index=df.index)
        loc_df.columns = [f"Node_{c}" for c in loc_df.columns]
        result.append(("Locations", loc_df))

        # Column 2: Fill values (possibly multi-day)
        first_fill = df.iloc[0, 2]
        if isinstance(first_fill, list) and len(first_fill) > 0:
            if isinstance(first_fill[0], list):
                num_days = len(first_fill)
                for d in range(num_days):
                    day_data = [row[d] for row in df[2]]
                    day_df = pd.DataFrame(day_data, index=df.index)
                    day_df.columns = [f"Bin_{c}" for c in day_df.columns]
                    result.append((f"Fill Values (Day {d + 1})", day_df))
            else:
                fill_df = pd.DataFrame(df[2].tolist(), index=df.index)
                fill_df.columns = [f"Bin_{c}" for c in fill_df.columns]
                result.append(("Fill Values", fill_df))
        else:
            fill_df = pd.DataFrame(df[2].tolist(), index=df.index)
            result.append(("Fill Values", fill_df))

        # Column 3: Max waste
        max_waste_data = df[3].tolist()
        if len(max_waste_data) > 0 and isinstance(max_waste_data[0], (list, np.ndarray)):
            mw_df = pd.DataFrame(max_waste_data, index=df.index)
        else:
            mw_df = pd.DataFrame(max_waste_data, index=df.index, columns=["Max Waste"])
        result.append(("Max Waste", mw_df))

        return result
    except Exception:
        return None


def _load_uploaded_file(uploaded_file: Any) -> Dict[str, pd.DataFrame]:
    """Parse an uploaded file into named DataFrames."""
    name = uploaded_file.name
    tables: Dict[str, pd.DataFrame] = {}

    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
        tables[f"{name} ({df.shape[0]}x{df.shape[1]})"] = df

    elif name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
        tables[f"{name} ({df.shape[0]}x{df.shape[1]})"] = df

    elif name.endswith(".pkl"):
        raw_data = pd.read_pickle(uploaded_file)
        dfs = _process_raw_to_dfs(raw_data)

        # Try VRPP heuristic on single-DF results
        if len(dfs) == 1:
            vrpp_result = _try_vrpp_split(dfs[0])
            if vrpp_result:
                for tname, tdf in vrpp_result:
                    tables[f"{tname} ({tdf.shape[0]}x{tdf.shape[1]})"] = tdf
                return tables

        for i, df in enumerate(dfs):
            key = f"Table {i + 1} ({df.shape[0]}x{df.shape[1]})"
            tables[key] = df

    return tables


# ---------------------------------------------------------------------------
# Chart rendering
# ---------------------------------------------------------------------------


def _render_chart(df: pd.DataFrame, chart_type: str, x_col: str, y_col: str) -> None:
    """Render the selected chart type."""
    if chart_type == "Heatmap":
        fig = create_heatmap_chart(df, title="Heatmap")
        st.plotly_chart(fig, width="stretch")
        return

    if not x_col or not y_col:
        st.warning("Select both X and Y axes.")
        return

    columns = df.columns.tolist()
    x_key = _resolve_column(columns, x_col)
    y_key = _resolve_column(columns, y_col)
    if x_key is None or y_key is None:
        st.error(f"Column not found: {x_col if x_key is None else y_col}")
        return

    x_data = df[x_key]
    y_data = df[y_key]

    if chart_type in ("Line Chart", "Area Chart"):
        sorted_df = pd.DataFrame({"x": x_data, "y": y_data}).sort_values("x")
        x_data, y_data = sorted_df["x"], sorted_df["y"]

    if chart_type == "Area Chart":
        fig = create_area_chart(x_data, y_data, x_label=x_col, y_label=y_col, title=f"Area: {y_col} vs {x_col}")
    elif chart_type == "Line Chart":
        fig = go.Figure(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="lines+markers",
                hovertemplate=f"{y_col}: %{{y:.4f}}<extra>%{{x}}</extra>",
            )
        )
        fig.update_layout(
            title=f"Line: {y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=400,
            **PLOTLY_LAYOUT_DEFAULTS,
        )
    elif chart_type == "Bar Chart":
        fig = go.Figure(
            go.Bar(
                x=x_data,
                y=y_data,
                marker_color=px.colors.qualitative.Set2,
                hovertemplate=f"{y_col}: %{{y:.4f}}<extra>%{{x}}</extra>",
            )
        )
        fig.update_layout(
            title=f"Bar: {y_col} vs {x_col}", xaxis_title=x_col, yaxis_title=y_col, height=400, **PLOTLY_LAYOUT_DEFAULTS
        )
    elif chart_type == "Scatter Plot":
        fig = go.Figure(
            go.Scatter(
                x=x_data,
                y=y_data,
                mode="markers",
                marker=dict(size=8, opacity=0.7),
                hovertemplate=f"{x_col}: %{{x}}<br>{y_col}: %{{y:.4f}}<extra></extra>",
            )
        )
        fig.update_layout(
            title=f"Scatter: {y_col} vs {x_col}",
            xaxis_title=x_col,
            yaxis_title=y_col,
            height=400,
            **PLOTLY_LAYOUT_DEFAULTS,
        )
    else:
        st.warning(f"Unknown chart type: {chart_type}")
        return

    st.plotly_chart(fig, width="stretch")


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


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def render_data_explorer() -> None:
    """Render the Data Explorer page."""
    st.title("Data Explorer")
    st.markdown("Upload and visualize CSV, XLSX, or PKL data files with interactive charts.")

    uploaded_file = st.file_uploader(
        "Upload a data file",
        type=["csv", "xlsx", "pkl"],
        help="Supports CSV, Excel, and Pickle files. PKL files with VRPP/WCVRP structure are auto-split.",
    )

    if uploaded_file is None:
        st.info("Upload a file to get started.")
        return

    # Load and cache in session state
    cache_key = f"data_explorer_{uploaded_file.name}_{uploaded_file.size}"
    if cache_key not in st.session_state:
        with st.spinner("Loading file..."):
            st.session_state[cache_key] = _load_uploaded_file(uploaded_file)

    tables: Dict[str, pd.DataFrame] = st.session_state[cache_key]

    if not tables:
        st.error("Could not extract any tables from this file.")
        return

    # Table selector
    table_names = list(tables.keys())
    selected_table = st.selectbox("Select Table / Slice", options=table_names, index=0)

    df = tables[selected_table]

    # Tabs: Raw Data | Visualization
    tab_data, tab_viz = st.tabs(["Raw Data", "Visualization"])

    with tab_data:
        st.markdown(f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns")
        st.dataframe(df, width="stretch", height=400)

        csv = df.to_csv(index=False)
        st.download_button("Download as CSV", csv, file_name=f"{selected_table}.csv", mime="text/csv")

    with tab_viz:
        col1, col2, col3 = st.columns(3)

        with col1:
            chart_type = st.selectbox("Chart Type", options=_CHART_TYPES, index=0, key="de_chart_type")

        str_columns = [str(c) for c in df.columns.tolist()]
        is_heatmap = chart_type == "Heatmap"

        with col2:
            x_col = st.selectbox("X Axis", options=str_columns, index=0, disabled=is_heatmap, key="de_x_axis")
        with col3:
            y_idx = min(1, len(str_columns) - 1)
            y_col = st.selectbox("Y Axis", options=str_columns, index=y_idx, disabled=is_heatmap, key="de_y_axis")

        _render_chart(df, chart_type, x_col, y_col)
