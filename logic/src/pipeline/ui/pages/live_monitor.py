"""
Live Monitor mode for the Streamlit dashboard.

Provides real-time log tailing and live metric visualization during
simulation execution. Ported from gui/src/windows/ts_results_window.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st

from logic.src.pipeline.ui.components.charts import create_simulation_metrics_chart
from logic.src.pipeline.ui.services.data_loader import (
    compute_daily_stats,
    discover_simulation_logs,
)
from logic.src.pipeline.ui.services.log_parser import (
    DayLogEntry,
    filter_entries,
    get_unique_policies,
    get_unique_samples,
    parse_log_file,
)
from logic.src.pipeline.ui.styles.kpi import create_kpi_row

_TARGET_METRICS = ["overflows", "kg", "ncol", "km", "kg/km", "profit", "cost", "kg_lost"]

_MAX_LOG_LINES = 200


# ---------------------------------------------------------------------------
# Log viewer
# ---------------------------------------------------------------------------


def _read_raw_lines(log_path: str, max_lines: int = _MAX_LOG_LINES) -> List[str]:
    """Read the last N lines from a log file."""
    try:
        path = Path(log_path)
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-max_lines:]
    except Exception:
        return []


def _render_log_viewer(log_path: str) -> None:
    """Render raw log output in a scrollable code block."""
    lines = _read_raw_lines(log_path)
    st.subheader("Raw Log Output")
    st.caption(f"Showing last {len(lines)} lines from `{Path(log_path).name}`")
    st.code("\n".join(lines) if lines else "(empty)", language="text", line_numbers=True)


# ---------------------------------------------------------------------------
# Live KPIs
# ---------------------------------------------------------------------------


def _render_live_kpis(entries: List[DayLogEntry], policy: Optional[str], sample_id: Optional[int]) -> None:
    """Render KPI cards from the latest entry matching filters."""
    filtered = filter_entries(entries, policy=policy, sample_id=sample_id)
    if not filtered:
        st.info("No matching entries yet.")
        return

    latest = max(filtered, key=lambda e: e.day)
    data = latest.data

    st.subheader(f"Latest Metrics (Day {latest.day})")

    kpis: Dict[str, Any] = {
        "Day": latest.day,
        "Profit": data.get("profit", 0),
        "Distance (km)": data.get("km", 0),
        "Waste (kg)": data.get("kg", 0),
        "Overflows": data.get("overflows", 0),
        "Collections": data.get("ncol", 0),
        "Efficiency (kg/km)": data.get("kg/km", 0),
        "Cost": data.get("cost", 0),
    }

    st.markdown(create_kpi_row(kpis), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Progress indicator
    n_entries = len(filtered)
    unique_days = len(set(e.day for e in filtered))
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Entries Parsed", n_entries)
    with col2:
        st.metric("Days Completed", unique_days)
    with col3:
        st.metric("Policy", latest.policy)


# ---------------------------------------------------------------------------
# Live chart
# ---------------------------------------------------------------------------


def _render_live_chart(entries: List[DayLogEntry], policy: Optional[str], selected_metric: str) -> None:
    """Render a live metric-over-time chart."""
    df = compute_daily_stats(entries, policy=policy)
    if df.empty:
        st.info("Not enough data for charting yet.")
        return

    available = [m for m in _TARGET_METRICS if f"{m}_mean" in df.columns]
    if selected_metric not in available:
        selected_metric = available[0] if available else None

    if selected_metric:
        st.subheader(f"Metric Over Time: {selected_metric}")
        fig = create_simulation_metrics_chart(df=df, metrics=[selected_metric], show_std=True)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------


def _render_sidebar_controls(
    policies: List[str],
    samples: List[int],
) -> Dict[str, Any]:
    """Render live monitor sidebar controls."""
    st.sidebar.markdown("---")
    st.sidebar.subheader("Live Filters")

    selected_policy = None
    if policies:
        policy_opt = st.sidebar.selectbox("Policy", options=["All"] + policies, index=0, key="lm_policy")
        if policy_opt != "All":
            selected_policy = policy_opt

    selected_sample = None
    if samples:
        sample_opts = ["All"] + [str(s) for s in samples]
        sample_opt = st.sidebar.selectbox("Sample", options=sample_opts, index=0, key="lm_sample")
        if sample_opt != "All":
            selected_sample = int(sample_opt)

    selected_metric = st.sidebar.selectbox("Metric to Plot", options=_TARGET_METRICS, index=5, key="lm_metric")

    return {
        "policy": selected_policy,
        "sample_id": selected_sample,
        "metric": selected_metric,
    }


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------


def render_live_monitor() -> None:
    """Render the Live Monitor page."""
    st.title("Live Simulation Monitor")
    st.markdown("Monitor simulation progress in real-time with auto-refresh.")

    logs = discover_simulation_logs()
    if not logs:
        st.info("No simulation logs found in `assets/output/`. Run a simulation with `python main.py test_sim`.")
        return

    log_names = [name for name, _ in logs]
    log_paths = {name: str(path) for name, path in logs}

    selected_log = st.sidebar.selectbox(
        "Select Log File",
        options=log_names,
        index=0,
        help="Choose a simulation log to monitor",
        key="lm_log_select",
    )

    if selected_log not in log_paths:
        st.error("Selected log file not found.")
        return

    log_path = log_paths[selected_log]

    # Parse entries fresh (no caching for live updates)
    entries = parse_log_file(Path(log_path))

    # Dynamic sidebar controls
    policies = get_unique_policies(entries) if entries else []
    samples = get_unique_samples(entries) if entries else []
    controls = _render_sidebar_controls(policies, samples)

    if not entries:
        st.warning("Log file is empty or contains no valid entries. Waiting for data...")
        _render_log_viewer(log_path)
        return

    # Layout: KPIs -> Chart -> Raw Log
    _render_live_kpis(entries, controls["policy"], controls["sample_id"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    _render_live_chart(entries, controls["policy"], controls["metric"])

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    with st.expander("Raw Log Output", expanded=False):
        _render_log_viewer(log_path)
