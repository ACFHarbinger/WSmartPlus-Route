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
import streamlit as st

from logic.src.constants import ROOT_DIR
from logic.src.ui.pages.simulation_summary_sections import (
    _render_daily_timeseries,
    _render_distribution_comparison,
    _render_kpi_overview,
    _render_metric_bar_chart,
    _render_pareto,
    _render_summary_table,
)

_DIST_PATTERN = re.compile(r"_(emp|gamma\d*|uniform)$", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data discovery & loading
# ---------------------------------------------------------------------------


def _discover_output_dirs() -> List[str]:
    """Find directories under assets/output/ that contain simulation JSONs."""

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


# Section renderers in simulation_summary_sections.py


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
