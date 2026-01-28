# Copyright (c) WSmart-Route. All rights reserved.
"""
MLOps Control Tower Dashboard - Main Entry Point.

A unified Streamlit interface for monitoring Deep Learning training
and visualizing VRP Simulation outputs.

Usage:
    streamlit run logic/src/pipeline/ui/app.py
"""

import time
from typing import List

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from logic.src.pipeline.ui.components.benchmark_charts import (
    create_benchmark_comparison_chart,
    create_latency_throughput_scatter,
)
from logic.src.pipeline.ui.components.charts import (
    create_simulation_metrics_chart,
    create_training_loss_chart,
)
from logic.src.pipeline.ui.components.maps import create_simulation_map
from logic.src.pipeline.ui.components.sidebar import (
    render_about_section,
    render_auto_refresh_toggle,
    render_mode_selector,
    render_simulation_controls,
    render_training_controls,
)
from logic.src.pipeline.ui.services.benchmark_loader import (
    get_unique_benchmarks,
    load_benchmark_data,
)
from logic.src.pipeline.ui.services.data_loader import (
    compute_daily_stats,
    discover_simulation_logs,
    discover_training_runs,
    load_multiple_training_runs,
    load_simulation_log_fresh,
)
from logic.src.pipeline.ui.services.log_parser import (
    filter_entries,
    get_day_range,
    get_unique_policies,
    get_unique_samples,
)
from logic.src.pipeline.ui.styles.styling import CUSTOM_CSS, create_kpi_row, get_page_config


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
        render_training_monitor()
    elif mode == "simulation":
        render_simulation_visualizer()
    else:
        render_benchmark_analysis()

    # Auto-refresh handling
    if auto_refresh:
        time.sleep(refresh_interval)
        st.rerun()


def render_training_monitor() -> None:
    """Render the Training Monitor mode."""
    st.title("📈 Training Monitor")
    st.markdown("Monitor and compare PyTorch Lightning training runs.")

    # Discover available runs
    runs = discover_training_runs()

    if not runs:
        st.warning(
            "⚠️ No training runs found in `logs/`.\n\n"
            "Start a training run with PyTorch Lightning to see metrics here."
        )
        return

    run_names = [name for name, _ in runs]

    # Get available metrics from first run
    first_run_data = load_multiple_training_runs([run_names[0]])
    available_metrics: List[str] = []
    if run_names[0] in first_run_data and not first_run_data[run_names[0]].empty:
        available_metrics = [col for col in first_run_data[run_names[0]].columns if col not in ["epoch", "step"]]

    if not available_metrics:
        available_metrics = ["train_loss", "val_loss", "val_cost"]

    # Render controls
    controls = render_training_controls(run_names, available_metrics)

    if not controls["selected_runs"]:
        st.info("👆 Select at least one run from the sidebar to view metrics.")
        return

    # Load selected runs
    runs_data = load_multiple_training_runs(controls["selected_runs"])

    # Create and display chart
    st.subheader("Loss Curves")

    fig = create_training_loss_chart(
        runs_data=runs_data,
        metric_y1=controls["primary_metric"],
        metric_y2=controls["secondary_metric"],
        x_axis=controls["x_axis"],
        smoothing=controls["smoothing"],
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show data tables in expander
    with st.expander("📊 View Raw Data"):
        for run_name, df in runs_data.items():
            st.markdown(f"**{run_name}**")
            st.dataframe(df.tail(20), use_container_width=True)


def render_simulation_visualizer() -> None:
    """Render the Simulation Digital Twin mode."""
    st.title("🗺️ Simulation Digital Twin")
    st.markdown("Visualize VRP simulation outputs with interactive maps.")

    # Discover available logs
    logs = discover_simulation_logs()

    if not logs:
        st.warning(
            "⚠️ No simulation logs found in `assets/output/`.\n\n"
            "Run a simulation with `python main.py test_sim` to generate logs."
        )
        return

    log_names = [name for name, _ in logs]
    log_paths = {name: str(path) for name, path in logs}

    # Initial placeholder values for controls
    policies: List[str] = []
    samples: List[int] = []
    day_range = (1, 1)

    # Get first log to populate controls
    if log_names:
        first_log_path = log_paths[log_names[0]]
        entries = load_simulation_log_fresh(first_log_path)
        if entries:
            policies = get_unique_policies(entries)
            samples = get_unique_samples(entries)
            day_range = get_day_range(entries)

    # Render sidebar controls
    controls = render_simulation_controls(
        available_logs=log_names,
        policies=policies,
        samples=samples,
        day_range=day_range,
    )

    # Load selected log
    if controls["selected_log"] not in log_paths:
        st.error("Selected log file not found.")
        return

    selected_path = log_paths[controls["selected_log"]]

    try:
        entries = load_simulation_log_fresh(selected_path)
    except Exception as e:
        st.error(f"Failed to load log file: {e}")
        return

    if not entries:
        st.warning("The selected log file is empty or contains no valid entries.")
        return

    # Update metadata after loading
    policies = get_unique_policies(entries)
    samples = get_unique_samples(entries)
    day_range = get_day_range(entries)

    # Filter entries
    filtered = filter_entries(
        entries,
        policy=controls["selected_policy"],
        sample_id=controls["selected_sample"],
        day=controls["selected_day"] if not controls["is_live"] else None,
    )

    if controls["is_live"]:
        # Get latest day
        latest_day = max(e.day for e in filtered) if filtered else day_range[1]
        filtered = filter_entries(filtered, day=latest_day)

    if not filtered:
        st.warning("No entries match the selected filters.")
        return

    # Get the entry to display (first one matching filters)
    display_entry = filtered[0]

    # KPI Dashboard
    if controls["show_stats"]:
        st.subheader("📊 Key Metrics")

        kpi_data = {
            "Day": display_entry.day,
            "Distance (km)": display_entry.data.get("km", 0),
            "Waste (kg)": display_entry.data.get("kg", 0),
            "Profit": display_entry.data.get("profit", 0),
            "Overflows": display_entry.data.get("overflows", 0),
        }

        st.markdown(create_kpi_row(kpi_data), unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Map Visualization
    st.subheader("🗺️ Route Map")

    tour = display_entry.data.get("tour", [])
    bin_states = display_entry.data.get("bin_state_c", [])
    bin_states_after = display_entry.data.get("bins_state_c_after", [])

    if not tour:
        st.warning("No tour data available for this entry.")
    else:
        # Determine served bins (those with bin_state_c_after = 0, meaning collected)
        served_indices = []
        for i, state in enumerate(bin_states_after):
            if state == 0 and i < len(bin_states) and bin_states[i] > 0:
                served_indices.append(i)

        sim_map = create_simulation_map(
            tour=tour,
            bin_states=bin_states,
            served_indices=served_indices,
            show_route=controls["show_route"],
        )

        st_folium(sim_map, width=None, height=500, returned_objects=[])

    # Statistics over time
    if controls["show_stats"]:
        with st.expander("📈 Metrics Over Time"):
            df = compute_daily_stats(entries, policy=controls["selected_policy"])

            if not df.empty:
                metrics_to_plot = ["profit", "km", "kg"]
                available_plot_metrics = [m for m in metrics_to_plot if f"{m}_mean" in df.columns]

                if available_plot_metrics:
                    fig = create_simulation_metrics_chart(
                        df=df,
                        metrics=available_plot_metrics,
                        show_std=True,
                    )
                    st.plotly_chart(fig, use_container_width=True)

    # Raw data view
    with st.expander("📋 View Entry Data"):
        st.json(display_entry.data)


def render_benchmark_analysis() -> None:
    """Render the Benchmark Analysis mode."""
    st.title("📊 Benchmark Analysis")
    st.markdown("Analyze performance metrics, latency, and throughput across different solvers and models.")

    # Load benchmark data
    df = load_benchmark_data()

    if df.empty:
        st.warning(
            "⚠️ No benchmark data found in `logs/benchmarks.jsonl`.\n\n"
            "Run `just benchmark` to generate some benchmarks."
        )
        return

    # Benchmark categories
    benchmarks = get_unique_benchmarks(df)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Benchmark Filters")
    selected_bench = st.sidebar.multiselect(
        "Select Benchmarks",
        options=benchmarks,
        default=benchmarks[:1] if benchmarks else [],
    )

    if not selected_bench:
        st.info("👆 Select at least one benchmark type from the sidebar.")
        return

    # Filter data
    filtered_df = df[df["benchmark"].isin(selected_bench)]

    # Analysis Tabs
    tab1, tab2, tab3 = st.tabs(["🏆 Performance Table", "📈 Comparison Charts", "⏱️ Latency & Throughput"])

    with tab1:
        st.subheader("Benchmark Results")
        st.dataframe(filtered_df, use_container_width=True)

    with tab2:
        st.subheader("Metric Comparison")
        metrics = [
            col for col in filtered_df.columns if col not in ["timestamp", "time_str", "benchmark", "device", "problem"]
        ]
        if metrics:
            selected_metric: str = st.selectbox("Select Metric to Visualize", options=metrics)
            if selected_metric:
                fig = create_benchmark_comparison_chart(
                    pd.DataFrame(filtered_df),
                    metric=selected_metric,
                    title=f"{selected_metric.capitalize()} Comparison",
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric metrics available for the selected benchmarks.")

    with tab3:
        st.subheader("Inference Efficiency")
        if "latency" in filtered_df.columns and "throughput" in filtered_df.columns:
            fig = create_latency_throughput_scatter(filtered_df)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Latency and throughput data only available for neural benchmarks.")


if __name__ == "__main__":
    main()
