from typing import Any, List

import streamlit as st

from logic.src.pipeline.ui.components.sidebar import render_simulation_controls
from logic.src.pipeline.ui.services.data_loader import (
    discover_simulation_logs,
    load_simulation_log_fresh,
)
from logic.src.pipeline.ui.services.log_parser import (
    get_day_range,
    get_unique_policies,
    get_unique_samples,
)

from .bins import render_bin_tab
from .charts import (
    render_metric_charts,
    render_policy_comparison,
    render_summary_statistics,
)
from .kpi import (
    render_cumulative_summary,
    render_kpi_dashboard,
    render_policy_info,
)
from .map import render_map_view
from .tour import render_raw_data_view, render_tour_details

# Local imports
from .utils import (
    filter_simulation_data,
    normalize_tour_points,
)


def render_simulation_visualizer() -> None:
    """Render the Simulation Digital Twin mode."""
    st.title("Simulation Digital Twin")
    st.markdown("Visualize VRP simulation outputs with interactive maps.")

    logs = discover_simulation_logs()
    if not logs:
        st.info("No simulation logs found in `assets/output/`. Run a simulation with `python main.py test_sim`.")
        return

    log_names = [name for name, _ in logs]
    log_paths = {name: str(path) for name, path in logs}

    # Initial defaults
    policies: List[str] = []
    samples: List[int] = []
    day_range = (1, 1)

    # Peek into first log
    if log_names:
        entries = load_simulation_log_fresh(log_paths[log_names[0]])
        if entries:
            policies = get_unique_policies(entries)
            samples = get_unique_samples(entries)
            day_range = get_day_range(entries)

    controls = render_simulation_controls(
        available_logs=log_names,
        policies=policies,
        samples=samples,
        day_range=day_range,
    )

    if controls["selected_log"] not in log_paths:
        st.error("Selected log file not found.")
        return

    with st.spinner("Loading simulation data..."):
        try:
            entries = load_simulation_log_fresh(log_paths[controls["selected_log"]])
        except Exception as e:
            st.error(f"Failed to load log: {e}")
            return

    if not entries:
        st.info("The selected log file is empty.")
        return

    # Filter and normalize
    day_range = get_day_range(entries)
    filtered = filter_simulation_data(entries, controls, day_range)
    if not filtered:
        st.info("No entries match filters.")
        return

    display_entry = filtered[0]
    tour = display_entry.data.get("tour", [])
    if tour:
        display_entry.data["tour"] = normalize_tour_points(tour)

    # 1-3. Main section
    if controls["show_stats"]:
        render_kpi_dashboard(display_entry, entries, controls)
        render_cumulative_summary(entries, controls)
    render_policy_info(display_entry)

    # 4. Map
    render_map_view(display_entry, controls)

    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # 5. Tabs
    tab_analysis, tab_bins, tab_tour = st.tabs(["Analysis", "Bins", "Tour & Data"])

    with tab_analysis:
        render_policy_comparison(entries, display_entry.day)
        if controls["show_stats"]:
            render_summary_statistics(entries, controls)
            st.subheader("Metrics Over Time")
            render_metric_charts(entries, controls)

    with tab_bins:
        render_bin_tab(display_entry)

    with tab_tour:
        render_tour_details(display_entry)
        render_raw_data_view(display_entry)
