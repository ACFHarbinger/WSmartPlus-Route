"""
Simulation Digital Twin mode for the Streamlit dashboard.
"""

import json
import os
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from logic.src.constants import ROOT_DIR
from logic.src.pipeline.ui.components.charts import (
    create_simulation_metrics_chart,
)
from logic.src.pipeline.ui.components.maps import create_simulation_map
from logic.src.pipeline.ui.components.sidebar import (
    render_simulation_controls,
)
from logic.src.pipeline.ui.services.data_loader import (
    compute_daily_stats,
    discover_simulation_logs,
    load_simulation_log_fresh,
)
from logic.src.pipeline.ui.services.log_parser import (
    filter_entries,
    get_day_range,
    get_unique_policies,
    get_unique_samples,
)
from logic.src.pipeline.ui.styles.styling import create_kpi_row
from streamlit_folium import st_folium


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
    bin_states_after = display_entry.data.get("bins_state_real_c_after", [])

    if not tour:
        st.warning("No tour data available for this entry.")
    else:
        dist_matrix = None

        # Load Matrix Strategy (Custom File)
        if controls.get("distance_strategy") == "load_matrix":
            selected_file = controls.get("selected_matrix_file")
            if selected_file:
                matrix_path = os.path.join(ROOT_DIR, "data", "wsr_simulator", "distance_matrix", selected_file)
                if os.path.isfile(matrix_path):
                    try:
                        if matrix_path.endswith((".xlsx", ".xls")):
                            df = pd.read_excel(matrix_path, header=None)
                        else:
                            # Default to CSV
                            df = pd.read_csv(matrix_path, header=None)

                        loaded_data = df.to_numpy()
                        # Assume first row and col are headers/indices and strip them
                        # Force conversion to float to handle potential string types in the sliced region
                        sliced_data = loaded_data[1:, 1:].astype(float)

                        # Apply Bin Index Subsetting if selected
                        selected_index_file = controls.get("selected_index_file")
                        if selected_index_file:
                            index_path = os.path.join(
                                ROOT_DIR, "data", "wsr_simulator", "bins_selection", selected_index_file
                            )
                            if os.path.isfile(index_path):
                                with open(index_path, "r") as f:
                                    indices_list = json.load(f)

                                sample_idx = 0
                                if isinstance(controls.get("selected_sample"), int):
                                    sample_idx = controls["selected_sample"]

                                if 0 <= sample_idx < len(indices_list):
                                    idx_list = indices_list[sample_idx]
                                    target_indices = np.array([-1] + idx_list) + 1

                                    # Robustness check: ensure indices conform to matrix bounds
                                    max_idx = sliced_data.shape[0] - 1
                                    valid_indices = target_indices[target_indices <= max_idx]

                                    if len(valid_indices) < len(target_indices):
                                        st.warning(f"Some indices were out of bounds and ignored. Max index: {max_idx}")

                                    sliced_data = sliced_data[valid_indices[:, None], valid_indices]
                                else:
                                    st.warning(
                                        f"Sample index {sample_idx} out of range for index file (len={len(indices_list)}). Using full matrix."
                                    )

                        dist_matrix = pd.DataFrame(sliced_data)

                    except Exception as e:
                        st.error(f"Failed to load matrix {selected_file}: {e}")

        # Determine served bins (those with bin_state_c_after = 0, meaning collected)
        served_indices = []
        for i, state in enumerate(bin_states_after):
            if state == 0 and i < len(bin_states) and bin_states[i] > 0:
                served_indices.append(i)

        sim_map = create_simulation_map(
            tour=tour,
            bin_states=bin_states,
            served_indices=served_indices,
            vehicle_id=0,
            show_route=controls["show_route"],
            zoom_start=13,
            distance_matrix=dist_matrix,
            dist_strategy=controls.get("distance_strategy", "hsd"),
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
