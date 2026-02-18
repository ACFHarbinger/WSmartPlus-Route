"""
Simulation Summary page for the Streamlit dashboard.

Loads simulation output JSON files (mean, std, full, daily) and provides:
- Policy comparison table with mean +/- std per metric
- Pareto front scatter for any two selected metrics
- Per-metric bar chart ranking across policies
- Per-distribution comparison for the same policy across distributions
- Daily time-series overlay per policy
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from logic.src.pipeline.ui.components.charts import (
    PLOTLY_LAYOUT_DEFAULTS,
    calculate_pareto_front,
    create_pareto_scatter_chart,
)
from logic.src.pipeline.ui.styles.kpi import create_kpi_row, format_number

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

_DIST_PATTERN = re.compile(r"_(emp|gamma\d*|uniform)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data discovery & loading
# ---------------------------------------------------------------------------


def _discover_output_dirs() -> List[str]:
    """Find directories under assets/output/ that contain simulation JSONs."""
    from logic.src.constants import ROOT_DIR

    output_root = os.path.join(ROOT_DIR, "assets", "output")
    dirs: List[str] = []
    if not os.path.isdir(output_root):
        return dirs

    for root, _subdirs, files in os.walk(output_root):
        json_files = [f for f in files if f.endswith(".json")]
        if json_files:
            rel = os.path.relpath(root, output_root)
            dirs.append(rel)

    return sorted(dirs)


def _load_json(path: str) -> Any:
    """Load and parse a JSON file, returning None on failure."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def _find_json_files(output_dir: str) -> Dict[str, str]:
    """
    Given a directory, discover the different JSON file types.

    Returns dict with keys like 'mean', 'std', 'full', 'daily_*' mapping
    to absolute file paths.
    """
    from logic.src.constants import ROOT_DIR

    abs_dir = os.path.join(ROOT_DIR, "assets", "output", output_dir)
    result: Dict[str, str] = {}

    if not os.path.isdir(abs_dir):
        return result

    for fname in sorted(os.listdir(abs_dir)):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(abs_dir, fname)
        lower = fname.lower()

        if "log_mean" in lower:
            result["mean"] = fpath
        elif "log_std" in lower:
            result["std"] = fpath
        elif "log_full" in lower:
            result["full"] = fpath
        elif lower.startswith("daily"):
            result[fname] = fpath

    return result


# ---------------------------------------------------------------------------
# Data processing
# ---------------------------------------------------------------------------


def _parse_policy_name(raw_name: str) -> Tuple[str, str]:
    """Split a raw policy key into (base_name, distribution)."""
    match = _DIST_PATTERN.search(raw_name)
    if match:
        dist = match.group(1).lower()
        base = raw_name[: match.start()].rstrip("_")
        return base, dist
    return raw_name, "unknown"


def _extract_distributions(mean_data: Dict[str, Any]) -> List[str]:
    """Extract sorted unique distributions from mean JSON keys."""
    dists_set = set()
    for key in mean_data:
        if isinstance(mean_data[key], dict):
            _, dist = _parse_policy_name(key)
            dists_set.add(dist)
    return sorted(dists_set)


def _build_summary_df(
    mean_data: Dict[str, Dict[str, float]],
    std_data: Optional[Dict[str, Dict[str, float]]] = None,
) -> pd.DataFrame:
    """
    Build a summary DataFrame from mean (and optional std) JSON data.

    Columns: Policy, Distribution, Policy_Key, metric1_mean, metric1_std, ...
    """
    rows: List[Dict[str, Any]] = []

    for policy_key, metrics in mean_data.items():
        if not isinstance(metrics, dict):
            continue
        base_name, dist = _parse_policy_name(policy_key)

        row: Dict[str, Any] = {
            "Policy": base_name,
            "Distribution": dist,
            "Policy_Key": policy_key,
        }

        for m, v in metrics.items():
            row[f"{m}_mean"] = v
            if std_data and policy_key in std_data:
                row[f"{m}_std"] = std_data[policy_key].get(m, 0.0)
            else:
                row[f"{m}_std"] = 0.0

        rows.append(row)

    return pd.DataFrame(rows)


def _build_daily_df(daily_data: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Flatten a daily JSON (policy -> {metric: [values per day]}) into a long DataFrame.

    Columns: Policy, Distribution, day, metric1, metric2, ...
    """
    rows: List[Dict[str, Any]] = []

    for policy_key, metrics in daily_data.items():
        if not isinstance(metrics, dict):
            continue

        base_name, dist = _parse_policy_name(policy_key)
        days = metrics.get("day", [])
        n_days = len(days)

        for d_idx in range(n_days):
            row: Dict[str, Any] = {
                "Policy": base_name,
                "Distribution": dist,
                "day": days[d_idx],
            }
            for m, vals in metrics.items():
                if m in ("day", "tour"):
                    continue
                if isinstance(vals, list) and d_idx < len(vals):
                    row[m] = vals[d_idx]
            rows.append(row)

    return pd.DataFrame(rows)


def _filter_by_dist(df: pd.DataFrame, dist_filter: str) -> pd.DataFrame:
    """Apply distribution filter to a DataFrame. Returns filtered copy."""
    if dist_filter == "All":
        return df
    return pd.DataFrame(df.loc[df["Distribution"] == dist_filter]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------


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
    st.dataframe(display_df, use_container_width=True, hide_index=True)

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
    st.plotly_chart(fig, use_container_width=True)


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

    st.plotly_chart(fig, use_container_width=True)

    # Show Pareto-optimal policies
    if pareto_indices:
        pareto_df = df.iloc[pareto_indices][["Policy", "Distribution", x_col, y_col]].reset_index(drop=True)
        pareto_df.columns = pd.Index(["Policy", "Distribution", x_metric, y_metric])
        st.markdown("**Pareto-optimal policies:**")
        st.dataframe(pareto_df, use_container_width=True, hide_index=True)


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
    st.plotly_chart(fig, use_container_width=True)

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
    st.dataframe(pivot_df, use_container_width=True, hide_index=True)


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
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def _render_sidebar_controls(
    available_dirs: List[str],
    distributions: List[str],
) -> Dict[str, Any]:
    """Render sidebar controls for Simulation Summary page."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Simulation Summary")

    selected_dir = st.sidebar.selectbox(
        "Output Directory",
        options=available_dirs,
        index=0,
        help="Select the simulation output directory to analyse",
    )

    dist_options = ["All"] + distributions
    dist_filter = st.sidebar.selectbox(
        "Distribution Filter",
        options=dist_options,
        index=0,
        help="Filter policies by distribution type (applies to all tabs except Distribution Comparison)",
    )

    return {
        "selected_dir": selected_dir,
        "dist_filter": dist_filter,
    }


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def render_simulation_summary() -> None:
    """Render the Simulation Summary page."""
    st.title("Simulation Summary")
    st.markdown("Compare policy performance across simulation experiments with statistics and Pareto analysis.")

    # Discover available output directories
    available_dirs = _discover_output_dirs()

    if not available_dirs:
        st.info("No simulation output found in `assets/output/`. Run a simulation first.")
        return

    # Use session state to track selected directory and avoid loading twice
    # First pass: load from first dir to get initial distributions
    selected_dir_key = "ss_selected_dir_prev"
    prev_dir = st.session_state.get(selected_dir_key, None)

    # Always load for the first available dir initially to populate sidebar
    init_dir = prev_dir if prev_dir in available_dirs else available_dirs[0]
    json_files = _find_json_files(init_dir)
    mean_data = _load_json(json_files["mean"]) if "mean" in json_files else None
    distributions: List[str] = _extract_distributions(mean_data) if isinstance(mean_data, dict) else []

    # Sidebar (needs distributions list)
    controls = _render_sidebar_controls(available_dirs, distributions)
    selected_dir = controls["selected_dir"]
    dist_filter = controls["dist_filter"]

    # Reload if directory changed from what we initially loaded
    if selected_dir != init_dir:
        json_files = _find_json_files(selected_dir)
        mean_data = _load_json(json_files["mean"]) if "mean" in json_files else None
        # Recompute distributions — sidebar won't update until next rerun,
        # but we store the dir so next render uses the right one.
        if isinstance(mean_data, dict):
            new_dists = _extract_distributions(mean_data)
            if new_dists != distributions:
                st.session_state[selected_dir_key] = selected_dir
                st.rerun()

    st.session_state[selected_dir_key] = selected_dir

    # Load std data
    std_data = _load_json(json_files["std"]) if "std" in json_files else None

    if not mean_data or not isinstance(mean_data, dict):
        st.warning(f"No `log_mean*.json` found in `{selected_dir}`.")
        return

    # Build summary DataFrame
    std_dict = std_data if isinstance(std_data, dict) else None
    summary_df = _build_summary_df(mean_data, std_dict)

    if summary_df.empty:
        st.error("Could not parse policy data from the selected directory.")
        return

    # KPI overview (respects distribution filter)
    _render_kpi_overview(summary_df, dist_filter)
    st.write("")

    # Section Selection (Persistent Tabs)
    tab_labels = ["Policy Table", "Metric Ranking", "Pareto Front", "Distribution Comparison", "Daily Trends"]
    selected_tab = st.segmented_control(
        "Simulation Summary View",
        options=tab_labels,
        default=tab_labels[0],
        key="ss_active_tab",
        label_visibility="collapsed",
    )
    st.write("")

    if selected_tab == "Policy Table":
        _render_summary_table(summary_df, dist_filter)

    elif selected_tab == "Metric Ranking":
        _render_metric_bar_chart(summary_df, dist_filter)

    elif selected_tab == "Pareto Front":
        _render_pareto(summary_df, dist_filter)

    elif selected_tab == "Distribution Comparison":
        _render_distribution_comparison(summary_df)

    elif selected_tab == "Daily Trends":
        # Find daily files and load
        daily_files = {k: v for k, v in json_files.items() if k.startswith("daily")}
        if daily_files:
            daily_names = list(daily_files.keys())
            selected_daily = st.selectbox(
                "Daily Data File",
                options=daily_names,
                index=0,
                key="ss_daily_file",
            )
            daily_data = _load_json(daily_files[selected_daily])
            if daily_data and isinstance(daily_data, dict):
                daily_df = _build_daily_df(daily_data)
                _render_daily_timeseries(daily_df, dist_filter)
            else:
                st.warning("Could not parse daily data.")
        else:
            st.info("No daily data files found in this directory.")
