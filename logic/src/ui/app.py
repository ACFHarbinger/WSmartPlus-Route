# Copyright (c) WSmart-Route. All rights reserved.
"""
MLOps Control Tower Dashboard - Main Entry Point.

A unified Streamlit interface for monitoring Deep Learning training
and visualizing VRP Simulation outputs.

Usage:
    streamlit run logic/src/pipeline/ui/app.py
"""

import time

import streamlit as st

from logic.src.ui.components.sidebar import (
    render_about_section,
    render_auto_refresh_toggle,
    render_mode_selector,
)
from logic.src.ui.pages import (
    render_benchmark_analysis,
    render_data_explorer,
    render_experiment_tracker,
    render_simulation_summary,
    render_simulation_visualizer,
    render_training_monitor,
)
from logic.src.ui.styles.colors import get_page_config
from logic.src.ui.styles.css import CUSTOM_CSS


def main() -> None:
    """Main entry point for the dashboard."""
    # Page configuration
    st.set_page_config(**get_page_config())

    # Apply custom CSS
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # Sidebar controls
    mode = render_mode_selector()
    auto_refresh, refresh_interval = render_auto_refresh_toggle()
    render_about_section()

    # Main content based on mode
    if mode == "training":
        with st.spinner("Loading Training Monitor..."):
            render_training_monitor()
    elif mode == "simulation":
        with st.spinner("Loading Simulation Digital Twin..."):
            render_simulation_visualizer()
    elif mode == "simulation_summary":
        with st.spinner("Loading Simulation Summary..."):
            render_simulation_summary()
    elif mode == "data_explorer":
        with st.spinner("Loading Data Explorer..."):
            render_data_explorer()
    elif mode == "experiment_tracker":
        with st.spinner("Loading Experiment Tracker..."):
            render_experiment_tracker()
    else:
        with st.spinner("Loading Benchmark Analysis..."):
            render_benchmark_analysis()

    # Auto-refresh handling
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


if __name__ == "__main__":
    main()
