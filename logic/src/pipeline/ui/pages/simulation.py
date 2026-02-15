"""
Simulation Digital Twin mode for the Streamlit dashboard.
"""

import contextlib
import json
import os
from typing import Any, Dict, List, Optional, Tuple

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from logic.src.constants import ROOT_DIR
from logic.src.pipeline.ui.components.charts import (
    create_radar_chart,
    create_simulation_metrics_chart,
    create_sparkline_svg,
    create_stacked_bar_chart,
)
from logic.src.pipeline.ui.components.maps import create_bin_heatmap, create_simulation_map
from logic.src.pipeline.ui.components.sidebar import (
    render_simulation_controls,
)
from logic.src.pipeline.ui.services.data_loader import (
    compute_cumulative_stats,
    compute_daily_stats,
    compute_day_deltas,
    compute_summary_statistics,
    discover_simulation_logs,
    get_metric_history,
    load_simulation_log_fresh,
)
from logic.src.pipeline.ui.services.log_parser import (
    filter_entries,
    get_day_range,
    get_unique_policies,
    get_unique_samples,
)
from logic.src.pipeline.ui.styles.kpi import create_kpi_row, create_kpi_row_with_deltas


def _normalize_tour_points(tour: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tour point keys: map 'lon' -> 'lng' for consistency with map components."""
    for point in tour:
        if "lon" in point and "lng" not in point:
            point["lng"] = point["lon"]
    return tour


def _filter_simulation_data(entries: List[Any], controls: Dict[str, Any], day_range: Tuple[int, int]) -> List[Any]:
    """Filter simulation entries based on controls."""
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

    return filtered


# Mapping from data keys to display labels for KPI deltas
_PRIMARY_KPI_MAP = {
    "profit": "Profit",
    "km": "Distance (km)",
    "kg": "Waste (kg)",
    "overflows": "Overflows",
}

_SECONDARY_KPI_MAP = {
    "ncol": "Collections",
    "kg_lost": "Waste Lost (kg)",
    "kg/km": "Efficiency (kg/km)",
    "cost": "Cost",
}


def _render_kpi_dashboard(
    display_entry: Any,
    entries: List[Any],
    controls: Dict[str, Any],
) -> None:
    """Render the Key Performance Indicators section with deltas and sparklines."""
    st.subheader("Key Metrics")

    data = display_entry.data
    current_day = display_entry.day

    # Compute day-over-day deltas
    deltas = compute_day_deltas(
        entries,
        current_day=current_day,
        policy=controls["selected_policy"],
        sample_id=controls["selected_sample"],
    )

    # Build sparklines from metric history
    sparklines: Dict[str, str] = {}
    for data_key, label in {**_PRIMARY_KPI_MAP, **_SECONDARY_KPI_MAP}.items():
        history = get_metric_history(
            entries,
            metric=data_key,
            policy=controls["selected_policy"],
            sample_id=controls["selected_sample"],
            last_n_days=7,
        )
        svg = create_sparkline_svg(history)
        if svg:
            sparklines[label] = svg

    # Row 1: Primary metrics with deltas
    primary_kpis: Dict[str, Tuple[Any, Optional[float]]] = {
        "Day": (display_entry.day, None),
        "Profit": (data.get("profit", 0), deltas.get("profit")),
        "Distance (km)": (data.get("km", 0), deltas.get("km")),
        "Waste (kg)": (data.get("kg", 0), deltas.get("kg")),
        "Overflows": (data.get("overflows", 0), deltas.get("overflows")),
    }
    st.markdown(create_kpi_row_with_deltas(primary_kpis, sparklines), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Secondary/efficiency metrics with deltas
    secondary_kpis: Dict[str, Tuple[Any, Optional[float]]] = {
        "Collections": (data.get("ncol", 0), deltas.get("ncol")),
        "Waste Lost (kg)": (data.get("kg_lost", 0), deltas.get("kg_lost")),
        "Efficiency (kg/km)": (data.get("kg/km", 0), deltas.get("kg/km")),
        "Cost": (data.get("cost", 0), deltas.get("cost")),
    }
    st.markdown(create_kpi_row_with_deltas(secondary_kpis, sparklines), unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


def _render_cumulative_summary(entries: List[Any], controls: Dict[str, Any]) -> None:
    """Render cumulative/aggregate statistics across all days."""
    cumulative = compute_cumulative_stats(
        entries,
        policy=controls["selected_policy"],
        sample_id=controls["selected_sample"],
    )
    if not cumulative:
        return

    with st.expander("Cumulative Summary (All Days)", expanded=False):
        st.markdown(create_kpi_row(cumulative), unsafe_allow_html=True)


def _render_policy_info(display_entry: Any) -> None:
    """Render policy configuration details parsed from the policy string."""
    with st.expander("Policy Configuration", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Policy", display_entry.policy)
        with col2:
            st.metric("Sample ID", display_entry.sample_id)
        with col3:
            st.metric("Day", display_entry.day)

        # Parse policy string for additional details
        policy_str = display_entry.policy
        parts = policy_str.split("_")

        details: Dict[str, Any] = {"Full Policy String": policy_str}

        known_policies = ["hgs", "alns", "gurobi", "tsp", "neural", "am", "ddam", "bcp"]
        known_selections = ["regular", "last_minute", "lookahead", "revenue", "service_level"]
        known_engines = ["gurobi", "pyvrp", "ortools"]

        detected_policy = next((p for p in known_policies if p in parts), None)
        detected_selection = next((s for s in known_selections if any(s in part for part in parts)), None)
        detected_engine = next((e for e in known_engines if e in parts), None)

        if detected_policy:
            details["Routing Policy"] = detected_policy
        if detected_selection:
            details["Selection Strategy"] = detected_selection
        if detected_engine:
            details["Solver Engine"] = detected_engine

        # Extract numeric parameters from known prefixes
        for part in parts:
            if part.startswith("lvl"):
                with contextlib.suppress(ValueError):
                    details["Selection Threshold"] = float(part[3:])
            elif part.startswith("gamma"):
                with contextlib.suppress(ValueError):
                    details["Gamma Parameter"] = float(part[5:])
            elif part.startswith("temp"):
                with contextlib.suppress(ValueError):
                    details["Temperature"] = float(part[4:])

        st.json(details)


def _load_custom_matrix(controls: Dict[str, Any]) -> Any:
    """Load a custom distance matrix if selected."""
    dist_matrix = None
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
                    # Force conversion to float to handle potential string types
                    sliced_data = loaded_data[1:, 1:].astype(float)

                    # Apply Bin Index Subsetting if selected
                    selected_index_file = controls.get("selected_index_file")
                    if selected_index_file:
                        index_path = os.path.join(
                            ROOT_DIR,
                            "data",
                            "wsr_simulator",
                            "bins_selection",
                            selected_index_file,
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

                                # Robustness check
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
    return dist_matrix


def _render_map_view(display_entry: Any, controls: Dict[str, Any]) -> None:
    """Render the route map."""
    st.subheader("Route Map")

    tour = display_entry.data.get("tour", [])
    bin_states = display_entry.data.get("bin_state_c", [])
    collected = display_entry.data.get("bin_state_collected", [])
    must_go: Optional[List[int]] = display_entry.data.get("must_go")
    all_bin_coords: Optional[List[Dict[str, Any]]] = display_entry.data.get("all_bin_coords")

    if not tour:
        st.info("No tour data available for this entry.")
        return

    dist_matrix = _load_custom_matrix(controls)

    sim_map = create_simulation_map(
        tour=tour,
        bin_states=bin_states,
        must_go=must_go,
        all_bin_coords=all_bin_coords,
        collected=collected if collected else None,
        vehicle_id=0,
        show_route=controls["show_route"],
        zoom_start=13,
        distance_matrix=dist_matrix,
        dist_strategy=controls.get("distance_strategy", "hsd"),
    )

    # Legend
    legend_html = """
    <div style="
        position: fixed; bottom: 30px; left: 30px; z-index: 1000;
        background: rgba(255,255,255,0.92); padding: 12px 16px;
        border-radius: 8px; border: 1px solid #ccc;
        font-size: 13px; line-height: 1.8;
        box-shadow: 0 2px 6px rgba(0,0,0,0.15);
    ">
        <b style="font-size: 14px;">Map Legend</b><br>
        <span style="color: #007bff;">&#9679;</span> Depot<br>
        <span style="color: #28a745;">&#9679;</span> Served (collected)<br>
        <span style="color: #fd7e14;">&#9679;</span> Must-Go (selected)<br>
        <span style="color: #dc3545;">&#9679;</span> Pending<br>
        <span style="color: #fd7e14;">&#9901;</span> Must-Go + Served
    </div>
    """
    sim_map.get_root().html.add_child(folium.Element(legend_html))  # type: ignore[attr-defined]

    st_folium(sim_map, width=None, height=500, returned_objects=[])


def _render_policy_comparison(entries: List[Any], selected_day: int) -> None:
    """Render radar chart comparing all policies for the selected day."""
    policies = get_unique_policies(entries)
    if len(policies) < 2:
        return

    st.subheader("Policy Comparison")

    radar_metrics = ["profit", "km", "kg", "overflows", "cost", "kg/km"]

    # Compute mean metric per policy for the selected day
    policy_metrics: Dict[str, Dict[str, float]] = {}
    for policy in policies:
        day_entries = filter_entries(entries, policy=policy, day=selected_day)
        if not day_entries:
            continue

        metrics: Dict[str, float] = {}
        for metric in radar_metrics:
            values = [e.data.get(metric, 0) for e in day_entries if metric in e.data]
            if values:
                metrics[metric] = float(np.mean(values))
        if metrics:
            policy_metrics[policy] = metrics

    if len(policy_metrics) < 2:
        return

    fig = create_radar_chart(policy_metrics, radar_metrics)
    st.plotly_chart(fig, use_container_width=True)


def _render_summary_statistics(entries: List[Any], controls: Dict[str, Any]) -> None:
    """Render descriptive statistics table (mean, std, min, max per metric)."""
    summary = compute_summary_statistics(entries, policy=controls["selected_policy"])
    if not summary:
        return

    st.subheader("Summary Statistics")

    rows = []
    for metric, stats in summary.items():
        rows.append(
            {
                "Metric": metric,
                "Mean": round(stats["mean"], 2),
                "Std": round(stats["std"], 2),
                "Min": round(stats["min"], 2),
                "Max": round(stats["max"], 2),
                "Total": round(stats["total"], 2),
            }
        )

    if rows:
        df = pd.DataFrame(rows).set_index("Metric")
        st.dataframe(df, use_container_width=True)


def _render_bin_heatmap(display_entry: Any) -> None:
    """Render bin fill level heatmap using the existing heatmap component."""
    tour = display_entry.data.get("tour", [])
    bin_states = display_entry.data.get("bin_state_c", [])

    if not tour or not bin_states:
        st.info("No bin state or tour data available for heatmap.")
        return

    # Build bin locations with matched fill levels from tour points
    bin_locations: List[Dict[str, Any]] = []
    matched_states: List[float] = []

    for point in tour:
        # Skip depot (id=0) and points without coordinates
        if "lat" not in point or "lng" not in point:
            continue
        try:
            bin_id = int(point["id"])
        except (ValueError, TypeError):
            continue
        if bin_id == 0:
            continue  # Skip depot

        # Map bin ID to fill level (try both 0-indexed and 1-indexed)
        fill = 50.0
        if 0 <= bin_id < len(bin_states):
            fill = bin_states[bin_id]
        elif 0 <= bin_id - 1 < len(bin_states):
            fill = bin_states[bin_id - 1]

        bin_locations.append(point)
        matched_states.append(fill)

    if not bin_locations:
        st.info("No bin locations found in tour data.")
        return

    heatmap = create_bin_heatmap(bin_locations, matched_states)
    st_folium(heatmap, width=None, height=400, returned_objects=[])


def _style_bin_table(df: pd.DataFrame) -> Any:
    """Apply conditional formatting to the bin state table."""
    return (
        df.style.background_gradient(
            subset=["Fill Before (%)"],
            cmap="RdYlGn_r",
            vmin=0,
            vmax=120,
        )
        .map(
            lambda v: "background-color: #ffebee; font-weight: bold" if v else "",
            subset=["Overflow"],
        )
        .map(
            lambda v: "color: #2e7d32; font-weight: bold" if v else "",
            subset=["Collected"],
        )
        .format(
            {
                "Fill Before (%)": "{:.1f}",
                "Fill After (%)": "{:.1f}",
                "Collected (kg)": "{:.2f}",
            }
        )
    )


def _render_bin_state_inspector(display_entry: Any) -> None:
    """Render detailed bin state inspection table."""
    data = display_entry.data
    bin_states_before = data.get("bin_state_c", [])
    bin_states_after = data.get("bins_state_real_c_after", [])
    bin_collected = data.get("bin_state_collected", [])
    must_go: Optional[List[int]] = data.get("must_go")

    if not bin_states_before:
        st.info("No bin state data available.")
        return

    n_bins = len(bin_states_before)
    must_go_set = set(must_go) if must_go else set()

    rows = []
    for i in range(n_bins):
        before = bin_states_before[i] if i < len(bin_states_before) else 0
        after = bin_states_after[i] if i < len(bin_states_after) else 0
        collected_amount = bin_collected[i] if i < len(bin_collected) else 0
        # must_go uses 1-indexed bin IDs
        was_selected = (i + 1) in must_go_set
        was_collected = collected_amount > 0
        is_overflow = before > 100

        rows.append(
            {
                "Bin ID": i + 1,
                "Fill Before (%)": round(before, 1),
                "Fill After (%)": round(after, 1),
                "Collected (kg)": round(collected_amount, 2),
                "Selected (must_go)": was_selected,
                "Collected": was_collected,
                "Overflow": is_overflow,
            }
        )

    df = pd.DataFrame(rows)

    # Summary statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Bins", n_bins)
    with col2:
        st.metric("Bins Selected", len(must_go_set))
    with col3:
        n_collected = sum(1 for r in rows if r["Collected"])
        st.metric("Bins Collected", n_collected)
    with col4:
        n_overflow = sum(1 for r in rows if r["Overflow"])
        st.metric("Bins Overflowing", n_overflow)

    # Filter controls
    filter_opt = st.radio(
        "Filter bins",
        ["All", "Selected (must_go)", "Collected", "Overflowing"],
        horizontal=True,
        key="bin_filter",
    )

    filtered_df = df
    if filter_opt == "Selected (must_go)":
        filtered_df = df[df["Selected (must_go)"]]
    elif filter_opt == "Collected":
        filtered_df = df[df["Collected"]]
    elif filter_opt == "Overflowing":
        filtered_df = df[df["Overflow"]]

    # Apply conditional formatting
    styled = _style_bin_table(pd.DataFrame(filtered_df))
    st.dataframe(styled, height=300, use_container_width=True)


def _render_collection_details(display_entry: Any) -> None:
    """Render collection details for the day."""
    data = display_entry.data
    bin_collected = data.get("bin_state_collected", [])
    bin_states_before = data.get("bin_state_c", [])
    must_go: Optional[List[int]] = data.get("must_go")

    if not bin_collected:
        st.info("No collection data available.")
        return

    # Collection summary metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Number of Collections", data.get("ncol", 0))
    with col2:
        total_kg = sum(c for c in bin_collected if c > 0)
        st.metric("Total Collected (kg)", f"{total_kg:.2f}")
    with col3:
        st.metric("Bins in must_go", len(must_go) if must_go else 0)

    # Per-bin collection breakdown (only bins that were collected)
    must_go_set = set(must_go) if must_go else set()
    collected_bins = []
    for i, amount in enumerate(bin_collected):
        if amount > 0:
            fill_before = bin_states_before[i] if i < len(bin_states_before) else 0
            collected_bins.append(
                {
                    "Bin ID": i + 1,
                    "Fill Before (%)": round(fill_before, 1),
                    "Amount Collected (kg)": round(amount, 2),
                    "Was in must_go": (i + 1) in must_go_set,
                }
            )

    if collected_bins:
        st.markdown("**Per-Bin Collection Breakdown:**")
        st.dataframe(pd.DataFrame(collected_bins), use_container_width=True)

        # Stacked bar chart: fill remaining vs collected
        categories = [str(b["Bin ID"]) for b in collected_bins]
        collected_values = [b["Amount Collected (kg)"] for b in collected_bins]
        remaining_values = [float(max(0, b["Fill Before (%)"] - b["Amount Collected (kg)"])) for b in collected_bins]

        fig = create_stacked_bar_chart(
            categories=categories,
            series={
                "Collected (kg)": collected_values,
                "Remaining (%)": remaining_values,
            },
            title="Collection Breakdown per Bin",
            x_label="Bin ID",
            y_label="Amount",
            colors=["#43a047", "#e8eaed"],
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No bins were collected on this day.")


def _render_tour_details(display_entry: Any) -> None:
    """Render tour sequence and leg details."""
    data = display_entry.data
    tour = data.get("tour", [])

    if not tour or len(tour) <= 1:
        st.info("No tour executed on this day (empty or depot-only tour).")
        return

    # Tour summary
    n_stops = sum(1 for p in tour if int(p.get("id", 0)) != 0)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Stops", n_stops)
    with col2:
        st.metric("Tour Length (nodes)", len(tour))
    with col3:
        st.metric("Distance (km)", f"{data.get('km', 0):.2f}")

    # Tour sequence table
    st.markdown("**Tour Sequence:**")
    tour_rows = []
    for i, point in enumerate(tour):
        point_id = point.get("id", "?")
        is_depot = str(point_id) == "0"
        tour_rows.append(
            {
                "Step": i,
                "Node ID": point_id,
                "Type": "depot" if is_depot else "bin",
                "Latitude": round(point["lat"], 6) if "lat" in point else "N/A",
                "Longitude": round(point["lng"], 6) if "lng" in point else "N/A",
            }
        )
    st.dataframe(pd.DataFrame(tour_rows), use_container_width=True)

    # must_go list display
    must_go = data.get("must_go", [])
    if must_go:
        st.markdown(f"**must_go Selection** ({len(must_go)} bins): `{must_go}`")


def _render_metric_charts(entries: List[Any], controls: Dict[str, Any]) -> None:
    """Render evaluation metrics charts with user-selectable metrics."""
    df = compute_daily_stats(entries, policy=controls["selected_policy"])

    if not df.empty:
        all_metrics = ["profit", "km", "kg", "overflows", "ncol", "kg_lost", "kg/km", "cost"]
        available_plot_metrics = [m for m in all_metrics if f"{m}_mean" in df.columns]

        if available_plot_metrics:
            selected_metrics = st.multiselect(
                "Select metrics to plot",
                options=available_plot_metrics,
                default=available_plot_metrics[:3],
                key="metrics_select",
            )

            if selected_metrics:
                fig = create_simulation_metrics_chart(
                    df=df,
                    metrics=selected_metrics,
                    show_std=True,
                )
                st.plotly_chart(fig, use_container_width=True)

        # Download button for daily stats
        csv = df.to_csv(index=False)
        st.download_button(
            "Download Daily Stats CSV",
            csv,
            file_name="daily_stats.csv",
            mime="text/csv",
            key="download_daily_stats",
        )


def _render_raw_data_view(display_entry: Any) -> None:
    """Render raw data view for debugging."""
    st.markdown(
        f"**Policy**: `{display_entry.policy}` | "
        f"**Sample**: `{display_entry.sample_id}` | "
        f"**Day**: `{display_entry.day}`"
    )
    st.json(display_entry.data)


def render_simulation_visualizer() -> None:
    """Render the Simulation Digital Twin mode."""
    st.title("Simulation Digital Twin")
    st.markdown("Visualize VRP simulation outputs with interactive maps.")

    # Discover available logs
    logs = discover_simulation_logs()

    if not logs:
        st.info(
            "No simulation logs found in `assets/output/`.\n\n"
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

    with st.spinner("Loading simulation data..."):
        try:
            entries = load_simulation_log_fresh(selected_path)
        except Exception as e:
            st.error(f"Failed to load log file: {e}")
            return

    if not entries:
        st.info("The selected log file is empty or contains no valid entries.")
        return

    # Update metadata after loading (in case user switched logs)
    policies = get_unique_policies(entries)
    samples = get_unique_samples(entries)
    day_range = get_day_range(entries)

    # Filter entries
    filtered = _filter_simulation_data(entries, controls, day_range)

    if not filtered:
        st.info("No entries match the selected filters. Try adjusting the sidebar controls.")
        return

    # Get the entry to display (first one matching filters)
    display_entry = filtered[0]

    # Normalize tour point keys (lon -> lng) for downstream components
    tour = display_entry.data.get("tour", [])
    if tour:
        display_entry.data["tour"] = _normalize_tour_points(tour)

    # Determine the selected day for policy comparison
    selected_day = display_entry.day

    # 1. KPI Dashboard (always visible)
    if controls["show_stats"]:
        _render_kpi_dashboard(display_entry, entries, controls)

    # 2. Cumulative Summary
    if controls["show_stats"]:
        _render_cumulative_summary(entries, controls)

    # 3. Policy Configuration
    _render_policy_info(display_entry)

    # 4. Map Visualization (always visible)
    _render_map_view(display_entry, controls)

    # Divider before tabs
    st.markdown('<hr class="section-divider">', unsafe_allow_html=True)

    # 5. Tabbed detail sections
    tab_analysis, tab_bins, tab_tour = st.tabs(["Analysis", "Bins", "Tour & Data"])

    with tab_analysis:
        # Policy comparison radar chart
        _render_policy_comparison(entries, selected_day)

        # Summary statistics
        if controls["show_stats"]:
            _render_summary_statistics(entries, controls)

        # Metrics over time
        if controls["show_stats"]:
            st.subheader("Metrics Over Time")
            _render_metric_charts(entries, controls)

    with tab_bins:
        # Bin Fill Heatmap
        st.subheader("Bin Fill Level Heatmap")
        _render_bin_heatmap(display_entry)

        # Bin State Inspector
        st.subheader("Bin State Inspector")
        _render_bin_state_inspector(display_entry)

        # Collection Details
        st.subheader("Collection Details")
        _render_collection_details(display_entry)

    with tab_tour:
        # Tour Details
        st.subheader("Tour Details")
        _render_tour_details(display_entry)

        # Raw data
        st.subheader("Raw Data (JSON)")
        _render_raw_data_view(display_entry)
