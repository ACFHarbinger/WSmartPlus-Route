"""
Data Explorer mode for the Streamlit dashboard.

Provides interactive file loading, table viewing, statistical profiling,
correlation analysis, and advanced charting for CSV, XLSX, PKL, JSON, and
JSONL data files.  Includes VRPP/WCVRP dataset splitting heuristics and
simulation-output analysis features (distribution filtering, Pareto front).

Ported & enhanced from ``gui/src/tabs/analysis/``.
"""

import json
import re
from collections import abc, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.src.pipeline.ui.components.charts import (
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

# Internal column prefixes used by simulation JSON pivoting
_META_COLUMNS = ("__Policy_Names__", "__Distributions__", "__File_IDs__")


# ---------------------------------------------------------------------------
# Data loading helpers
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
                slice_data = np.squeeze(raw_data[i])
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


# ---------------------------------------------------------------------------
# JSON / JSONL loading  (ported from gui output_analysis/engine.py)
# ---------------------------------------------------------------------------

_DIST_PATTERN = re.compile(r"_(emp|gamma\d+|uniform)\b", re.IGNORECASE)


def _pivot_json_data(
    data: Dict[str, Any],
    file_id: str = "",
) -> Dict[str, List[Any]]:
    """Flatten nested simulation JSON into columnar form with metadata."""
    metrics: Dict[str, List[Any]] = defaultdict(list)
    policy_names: List[str] = []
    distributions: List[str] = []
    file_ids: List[str] = []

    for policy, results in data.items():
        if not isinstance(results, dict):
            continue

        match = _DIST_PATTERN.search(policy)
        if match:
            dist = match.group(1).lower()
            base_name = policy[: match.start()] + policy[match.end() :]
            base_name = base_name.rstrip("_")
        else:
            dist = "unknown"
            base_name = policy

        distributions.append(dist)
        policy_names.append(base_name)
        file_ids.append(file_id)

        for metric, value in results.items():
            metrics[metric].append(value)

    metrics["__Policy_Names__"] = policy_names
    metrics["__Distributions__"] = distributions
    metrics["__File_IDs__"] = file_ids
    return dict(metrics)


def _load_json_file(uploaded_file: Any) -> Dict[str, pd.DataFrame]:
    """Load a JSON simulation results file."""
    tables: Dict[str, pd.DataFrame] = {}
    name = uploaded_file.name
    try:
        content = json.loads(uploaded_file.getvalue())
    except Exception:
        return tables

    if isinstance(content, dict):
        pivoted = _pivot_json_data(content, file_id=name)
        if pivoted and pivoted.get("__Policy_Names__"):
            df = pd.DataFrame(pivoted)
            tables[f"{name} — Simulation Results ({df.shape[0]}x{df.shape[1]})"] = df
        else:
            # Flat dict → single-row DataFrame
            df = pd.json_normalize(content)
            tables[f"{name} ({df.shape[0]}x{df.shape[1]})"] = df
    elif isinstance(content, list):
        df = pd.json_normalize(content)
        tables[f"{name} ({df.shape[0]}x{df.shape[1]})"] = df

    return tables


def _load_jsonl_file(uploaded_file: Any) -> Dict[str, pd.DataFrame]:
    """Load a JSONL file (one JSON object per line)."""
    tables: Dict[str, pd.DataFrame] = {}
    name = uploaded_file.name
    try:
        lines = uploaded_file.getvalue().decode("utf-8").strip().splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    except Exception:
        return tables

    if records:
        df = pd.json_normalize(records)
        tables[f"{name} ({df.shape[0]}x{df.shape[1]})"] = df

    return tables


def _load_npz_file(uploaded_file: Any) -> Dict[str, pd.DataFrame]:
    """Load a .npz file into named DataFrames, one per array key."""
    tables: Dict[str, pd.DataFrame] = {}
    data = np.load(uploaded_file)
    for key in data.files:
        arr = data[key]
        if arr.ndim == 1:
            df = pd.DataFrame(arr, columns=[key])
        elif arr.ndim == 2:
            df = pd.DataFrame(arr)
        elif arr.ndim == 3:
            # (samples, days, bins) → one table per sample dimension
            for i in range(arr.shape[0]):
                slice_df = pd.DataFrame(arr[i])
                tables[f"{key} [sample {i}] ({slice_df.shape[0]}x{slice_df.shape[1]})"] = slice_df
            continue
        else:
            df = pd.DataFrame(arr.reshape(arr.shape[0], -1))
        tables[f"{key} ({df.shape[0]}x{df.shape[1]})"] = df
    return tables


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

    elif name.endswith(".json"):
        return _load_json_file(uploaded_file)

    elif name.endswith(".jsonl"):
        return _load_jsonl_file(uploaded_file)

    elif name.endswith(".npz"):
        return _load_npz_file(uploaded_file)

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
# Column helpers
# ---------------------------------------------------------------------------


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


def _safe_nunique(s: pd.Series) -> int:
    """Safely calculate unique values, falling back to string if unhashable."""
    try:
        return int(s.nunique())
    except TypeError:
        # Fallback for unhashable types like lists
        return int(s.astype(str).nunique())


def _has_distribution_meta(df: pd.DataFrame) -> bool:
    """Check whether DataFrame contains simulation distribution metadata."""
    return "__Distributions__" in df.columns


def _unique_distributions(df: pd.DataFrame) -> List[str]:
    """Return sorted unique distribution values."""
    if "__Distributions__" not in df.columns:
        return []
    return sorted(df["__Distributions__"].dropna().unique().tolist())


# ---------------------------------------------------------------------------
# Tab renderers
# ---------------------------------------------------------------------------


def _render_raw_data_tab(
    df: pd.DataFrame,
    selected_table: str,
    visible_columns: Optional[List[str]],
    row_limit: int,
    precision: int,
) -> None:
    """Render the Raw Data tab with optional column/row filtering."""
    display_df = df

    # Apply column filter
    if visible_columns is not None and len(visible_columns) < len(df.columns):
        resolved = [c for c in df.columns if str(c) in visible_columns]
        if resolved:
            display_df = display_df[resolved]

    # Apply row limit
    if row_limit < len(display_df):
        display_df = display_df.head(row_limit)

    st.markdown(f"**Shape**: {df.shape[0]} rows x {df.shape[1]} columns")
    if row_limit < len(df):
        st.caption(f"Showing first {row_limit} of {len(df)} rows")

    col_format = {c: f"{{:.{precision}f}}" for c in display_df.select_dtypes(include=["number"]).columns}
    st.dataframe(display_df.style.format(col_format, na_rep="—"), use_container_width=True, height=400)

    csv = df.to_csv(index=False)
    st.download_button("Download as CSV", csv, file_name=f"{selected_table}.csv", mime="text/csv")


def _render_statistics_tab(df: pd.DataFrame) -> None:
    """Render the Statistics tab with data profiling and descriptive stats."""

    # --- Data profile summary ---
    st.subheader("Data Profile")
    profile_cols = st.columns(4)
    profile_cols[0].metric("Rows", f"{df.shape[0]:,}")
    profile_cols[1].metric("Columns", f"{df.shape[1]:,}")
    mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    profile_cols[2].metric("Memory", f"{mem_mb:.2f} MB")
    n_numeric = len(df.select_dtypes(include=["number"]).columns)
    profile_cols[3].metric("Numeric Cols", str(n_numeric))

    # --- Column info table ---
    st.subheader("Column Information")
    col_info = pd.DataFrame(
        {
            "Column": [str(c) for c in df.columns],
            "Type": [str(df[c].dtype) for c in df.columns],
            "Non-Null": [int(df[c].notna().sum()) for c in df.columns],
            "Null": [int(df[c].isna().sum()) for c in df.columns],
            "Unique": [_safe_nunique(df[c]) for c in df.columns],
        }
    )
    st.dataframe(col_info, use_container_width=True, hide_index=True)

    # --- Descriptive statistics ---
    numeric_df = df.select_dtypes(include=["number"])
    if not numeric_df.empty:
        st.subheader("Descriptive Statistics")
        desc = numeric_df.describe().T
        desc = desc.set_index(pd.Index([str(c) for c in desc.index]))
        st.dataframe(desc.style.format("{:.4f}"), use_container_width=True)
    else:
        st.info("No numeric columns for descriptive statistics.")


def _render_correlation_tab(df: pd.DataFrame) -> None:
    """Render the Correlation tab with matrix heatmap and pair scatter."""

    numeric_cols = _numeric_columns(df)
    if len(numeric_cols) < 2:
        st.info("Need at least 2 numeric columns for correlation analysis.")
        return

    # --- Correlation matrix heatmap ---
    st.subheader("Correlation Matrix")
    fig = create_correlation_matrix_chart(df, title="Pairwise Pearson Correlation")
    st.plotly_chart(fig, use_container_width=True)

    # --- Pair scatter ---
    st.subheader("Column Pair Detail")
    pair_col1, pair_col2 = st.columns(2)
    with pair_col1:
        col_a = st.selectbox("Column A", options=numeric_cols, index=0, key="corr_col_a")
    with pair_col2:
        idx_b = min(1, len(numeric_cols) - 1)
        col_b = st.selectbox("Column B", options=numeric_cols, index=idx_b, key="corr_col_b")

    col_a_key = _resolve_column(df.columns.tolist(), col_a)
    col_b_key = _resolve_column(df.columns.tolist(), col_b)
    if col_a_key is not None and col_b_key is not None:
        show_trendline = st.checkbox("Show trend line (OLS)", value=False, key="corr_trend")
        trend = "ols" if show_trendline else None
        fig_pair = px.scatter(
            df,
            x=col_a_key,
            y=col_b_key,
            trendline=trend,
            title=f"{col_a} vs {col_b}",
            opacity=0.7,
        )
        fig_pair.update_layout(height=400, **PLOTLY_LAYOUT_DEFAULTS)
        st.plotly_chart(fig_pair, use_container_width=True)

        # Correlation coefficient
        corr_val = df[col_a_key].corr(df[col_b_key])
        st.markdown(f"**Pearson r** = `{corr_val:.4f}`")


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
            st.plotly_chart(fig, use_container_width=True)
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

    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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
    st.plotly_chart(fig, use_container_width=True)


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
        st.plotly_chart(create_heatmap_chart(df, title="Heatmap"), use_container_width=True)
    elif chart_type == "Correlation Matrix":
        st.plotly_chart(create_correlation_matrix_chart(df), use_container_width=True)
    elif chart_type == "Histogram":
        y_col = local_vars.get("y_col", "")
        col_key = _resolve_column(columns, y_col)
        if col_key is not None:
            nbins = extra_opts.get("nbins", 30)
            st.plotly_chart(
                create_histogram_chart(df[col_key], nbins=nbins, title=f"Histogram: {y_col}"),
                use_container_width=True,
            )
        else:
            st.warning("Select a valid column.")
    elif chart_type == "Box Plot":
        y_cols = local_vars.get("y_cols", [])
        if y_cols:
            resolved = [str(c) for c in columns if str(c) in y_cols]
            st.plotly_chart(
                create_box_plot_chart(df, resolved, title="Box Plot: Distribution Comparison"),
                use_container_width=True,
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
# Sidebar controls
# ---------------------------------------------------------------------------


def _render_sidebar_controls(df: pd.DataFrame) -> Dict[str, Any]:
    """Render Data Explorer sidebar controls. Returns settings dict."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Explorer Settings")

    str_columns = [str(c) for c in df.columns.tolist()]

    visible_columns = st.sidebar.multiselect(
        "Visible Columns",
        options=str_columns,
        default=str_columns,
        help="Select columns to display in Raw Data tab",
    )

    row_limit = st.sidebar.slider(
        "Row Display Limit",
        min_value=min(1, len(df)),
        max_value=max(len(df), 1),
        value=min(len(df), 1000),
        step=max(1, min(10, len(df) // 10)),
        help="Limit rows shown in Raw Data tab",
    )

    precision = st.sidebar.slider(
        "Decimal Precision",
        min_value=0,
        max_value=8,
        value=4,
        help="Decimal places for numeric formatting",
    )

    return {
        "visible_columns": visible_columns,
        "row_limit": row_limit,
        "precision": precision,
    }


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def render_data_explorer() -> None:
    """Render the Data Explorer page."""
    st.title("Data Explorer")
    st.markdown(
        "Upload and analyse CSV, XLSX, NPZ, PKL, JSON, or JSONL data files "
        "with interactive charts, statistics, and correlation analysis."
    )

    uploaded_file = st.file_uploader(
        "Upload a data file",
        type=["csv", "xlsx", "npz", "pkl", "json", "jsonl"],
        help=(
            "Supports CSV, Excel, NumPy (.npz), Pickle, JSON, and JSONL files. "
            "NPZ simulation datasets display each named array as a table. "
            "JSON simulation results are pivoted with distribution metadata."
        ),
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

    # Sidebar controls
    sidebar_opts = _render_sidebar_controls(df)
    st.write("")

    # Section Selection (Persistent Tabs)
    tab_labels = ["Raw Data", "Statistics", "Correlation", "Visualization"]
    selected_tab = st.segmented_control(
        "Data Explorer View",
        options=tab_labels,
        default=tab_labels[0],
        key="de_active_tab",
        label_visibility="collapsed",
    )
    st.write("")

    if selected_tab == "Raw Data":
        _render_raw_data_tab(
            df,
            selected_table,
            sidebar_opts["visible_columns"],
            sidebar_opts["row_limit"],
            sidebar_opts["precision"],
        )

    elif selected_tab == "Statistics":
        _render_statistics_tab(df)

    elif selected_tab == "Correlation":
        _render_correlation_tab(df)

    elif selected_tab == "Visualization":
        _render_visualization_tab(df)
