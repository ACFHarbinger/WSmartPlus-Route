# Copyright (c) WSmart-Route. All rights reserved.
"""
Sidebar control panel widgets for the dashboard.

Provides reusable sidebar components for mode selection and controls.
"""

from typing import Any, Dict, List, Tuple

import streamlit as st


def render_mode_selector() -> str:
    """
    Render the main mode selector in the sidebar.

    Returns:
        Selected mode: "training" or "simulation"
    """
    st.sidebar.title("🎛️ Control Tower")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "📊 Dashboard Mode",
        options=["Training Monitor", "Simulation Digital Twin", "Benchmark Analysis"],
        index=1,  # Default to Simulation
        help="Switch between training metrics, simulation visualization, and benchmark analysis",
    )

    if mode == "Training Monitor":
        return "training"
    elif mode == "Simulation Digital Twin":
        return "simulation"
    else:
        return "benchmark"


def render_auto_refresh_toggle() -> Tuple[bool, int]:
    """
    Render auto-refresh controls.

    Returns:
        Tuple of (is_enabled, refresh_interval_seconds)
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🔄 Auto-Refresh")

    enabled = st.sidebar.checkbox("Enable Auto-Refresh", value=False)
    interval = st.sidebar.slider(
        "Refresh Interval (seconds)",
        min_value=2,
        max_value=30,
        value=5,
        disabled=not enabled,
    )

    return enabled, interval


def render_training_controls(
    available_runs: List[str],
    available_metrics: List[str],
) -> Dict[str, Any]:
    """
    Render controls for training monitor mode.

    Args:
        available_runs: List of available training run names.
        available_metrics: List of available metric names.

    Returns:
        Dict with selected runs, metrics, and settings.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("📈 Training Controls")

    # Multi-select for runs
    selected_runs = st.sidebar.multiselect(
        "Select Runs to Compare",
        options=available_runs,
        default=available_runs[:2] if len(available_runs) >= 2 else available_runs,
        help="Select multiple runs to compare their metrics",
    )

    # Primary metric
    primary_metric = st.sidebar.selectbox(
        "Primary Metric (Left Y-Axis)",
        options=available_metrics,
        index=0 if available_metrics else 0,
    )

    # Secondary metric
    use_secondary = st.sidebar.checkbox("Show Secondary Metric", value=False)
    secondary_metric = None
    if use_secondary:
        remaining_metrics = [m for m in available_metrics if m != primary_metric]
        if remaining_metrics:
            secondary_metric = st.sidebar.selectbox(
                "Secondary Metric (Right Y-Axis)",
                options=remaining_metrics,
                index=0,
            )

    # X-axis selection
    x_axis = st.sidebar.radio(
        "X-Axis",
        options=["epoch", "step"],
        index=0,
        horizontal=True,
    )

    # Smoothing
    smoothing = st.sidebar.slider(
        "Smoothing (Moving Average)",
        min_value=1,
        max_value=50,
        value=1,
        help="Apply moving average smoothing to curves",
    )

    return {
        "selected_runs": selected_runs,
        "primary_metric": primary_metric,
        "secondary_metric": secondary_metric,
        "x_axis": x_axis,
        "smoothing": smoothing,
    }


def render_simulation_controls(
    available_logs: List[str],
    policies: List[str],
    samples: List[int],
    day_range: Tuple[int, int],
) -> Dict[str, Any]:
    """
    Render controls for simulation visualizer mode.

    Args:
        available_logs: List of available log file names.
        policies: List of policy names in the log.
        samples: List of sample IDs.
        day_range: (min_day, max_day) tuple.

    Returns:
        Dict with selected options.
    """
    st.sidebar.markdown("---")
    st.sidebar.subheader("🗺️ Simulation Controls")

    # Log file selector
    selected_log = st.sidebar.selectbox(
        "Select Log File",
        options=available_logs,
        index=0 if available_logs else 0,
        help="Choose a simulation output file to visualize",
    )

    # Policy filter
    selected_policy = None
    if policies:
        selected_policy = st.sidebar.selectbox(
            "Filter by Policy",
            options=["All"] + policies,
            index=0,
        )
        if selected_policy == "All":
            selected_policy = None

    # Sample selector
    selected_sample = None
    if samples:
        sample_options = ["All"] + [str(s) for s in samples]
        selected_sample_str = st.sidebar.selectbox(
            "Select Sample",
            options=sample_options,
            index=0,
        )
        if selected_sample_str != "All":
            selected_sample = int(selected_sample_str)

    # Playback mode
    st.sidebar.markdown("---")
    st.sidebar.subheader("▶️ Playback")

    playback_mode = st.sidebar.radio(
        "Mode",
        options=["Live (Latest)", "Replay (Slider)"],
        index=1,
        horizontal=True,
    )

    is_live = playback_mode == "Live (Latest)"

    # Day slider for replay mode
    selected_day = day_range[1]  # Default to latest
    if not is_live and day_range[0] < day_range[1]:
        selected_day = st.sidebar.slider(
            "Day",
            min_value=day_range[0],
            max_value=day_range[1],
            value=day_range[1],
        )

    # Display options
    st.sidebar.markdown("---")
    st.sidebar.subheader("👁️ Display Options")

    show_route = st.sidebar.checkbox("Show Route Lines", value=True)
    show_stats = st.sidebar.checkbox("Show Statistics", value=True)

    display_options = {
        "Haversine (Spherical)": "hsd",
        "Geodesic (WGS84)": "gdsc",
        "Euclidean (Planar)": "ogd",
        "Load Matrix (Custom)": "load_matrix",
    }

    strategy_label = st.sidebar.selectbox(
        "Distance Strategy",
        options=list(display_options.keys()),
        index=3,  # Default to Load Matrix if available, or Haversine
        help="Choose method for calculating edge distances.\n'Load Matrix' allows selecting a custom file.",
    )

    distance_strategy = display_options[strategy_label]

    selected_matrix_file = None
    selected_index_file = None
    if distance_strategy == "load_matrix":
        import os

        from logic.src.constants import ROOT_DIR

        matrix_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator", "distance_matrix")

        # Helper to recursively find files
        matrix_files = []
        if os.path.exists(matrix_dir):
            for root, _dirs, files in os.walk(matrix_dir):
                for file in files:
                    if file.endswith((".csv", ".xlsx", ".txt")):
                        rel_path = os.path.relpath(os.path.join(root, file), matrix_dir)
                        matrix_files.append(rel_path)

        selected_matrix_file = st.sidebar.selectbox(
            "Select Matrix File",
            options=sorted(matrix_files),
            index=0 if matrix_files else 0,
            help="Select the distance matrix file to load.",
        )

        # Bin Index File Selector
        bins_selection_dir = os.path.join(ROOT_DIR, "data", "wsr_simulator", "bins_selection")
        index_files = []
        if os.path.exists(bins_selection_dir):
            for root, _dirs, files in os.walk(bins_selection_dir):
                for file in files:
                    if file.endswith(".json"):
                        rel_path = os.path.relpath(os.path.join(root, file), bins_selection_dir)
                        index_files.append(rel_path)

        selected_index_file = st.sidebar.selectbox(
            "Select Bin Index File (Optional)",
            options=["None"] + sorted(index_files),
            index=0,
            help="Select a JSON file containing bin indices to subset the matrix.",
        )
        if selected_index_file == "None":
            selected_index_file = None

    return {
        "selected_log": selected_log,
        "selected_policy": selected_policy,
        "selected_sample": selected_sample,
        "is_live": is_live,
        "selected_day": selected_day,
        "show_route": show_route,
        "show_stats": show_stats,
        "distance_strategy": distance_strategy,
        "selected_matrix_file": selected_matrix_file,
        "selected_index_file": selected_index_file,
    }


def render_about_section() -> None:
    """Render an about section at the bottom of the sidebar."""
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        <div style="text-align: center; color: #666; font-size: 12px;">
            <p>WSmart+ Route<br>MLOps Control Tower</p>
            <p>v1.0.0</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
