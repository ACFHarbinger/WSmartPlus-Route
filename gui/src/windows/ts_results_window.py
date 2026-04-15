"""
Simulation results visualization window.

This module provides a specialized window for displaying real-time and
summary statistics from simulation experiments. It acts as an orchestrator
for specialized sub-widgets and data managers.
"""

import os
import webbrowser
from typing import Optional

import folium
import logic.src.constants as udef
from logic.src.data.datasets import (
    NumpyDictDataset,
    NumpyPickleDataset,
    PandasExcelDataset,
    SimulationDataset,
)
from PySide6.QtCore import QMutex, Qt, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QLabel,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ..constants import TARGET_METRICS
from ..helpers import ChartWorker, FileTailerWorker
from .ts_results import LiveDashboardTab, SimulationDataManager, SummaryStatisticsTab


class SimulationResultsWindow(QWidget):
    """
    Orchestrator window for simulation results visualization.
    """

    start_chart_processing = Signal(str)

    def __init__(self, policy_names, log_path=None, dataset_path=None):
        """
        Initialize the simulation results orchestration window.

        Args:
            policy_names (list[str]): List of policy identifiers used to track results.
            log_path (str, optional): Absolute path to the simulation log file to tail.
            dataset_path (str, optional): Path to the simulation dataset file (.npz/.pkl/.xlsx)
                used to load bin coordinates for map visualization.
        """
        super().__init__()
        self.setWindowTitle("Simulation Results")
        self.setWindowFlags(self.windowFlags() | Qt.Window)  # type: ignore[attr-defined]
        self.resize(1200, 800)

        self.data_mutex = QMutex()
        self.policy_names = policy_names

        # Data Management
        self.data_manager = SimulationDataManager(policy_names)
        self.historical_routes = {}  # key -> day -> routes

        # Load simulation dataset for bin coordinates
        self.dataset: Optional[SimulationDataset] = self._load_dataset(dataset_path)

        # UI Initialization
        self.setup_ui()

        # Worker Threads
        self.setup_workers(log_path)

    @staticmethod
    def _load_dataset(dataset_path: Optional[str]) -> Optional[SimulationDataset]:
        """Load a simulation dataset from the given path for coordinate access."""
        if not dataset_path:
            return None
        # Resolve relative paths against ROOT_DIR
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.join(udef.ROOT_DIR, dataset_path)
        if not os.path.isfile(dataset_path):
            return None
        if dataset_path.endswith(".pkl"):
            return NumpyPickleDataset.load(dataset_path)
        if dataset_path.endswith(".xlsx"):
            return PandasExcelDataset.load(dataset_path)
        return NumpyDictDataset.load(dataset_path)

    def setup_ui(self):
        """Build the main window layout."""
        layout = QVBoxLayout(self)

        self.status_label = QLabel("Waiting for simulation data...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold; margin: 5px;")
        layout.addWidget(self.status_label)

        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)

        # Tab 1: Live Tracking
        self.dashboard_tab = LiveDashboardTab(self.policy_names)
        self.dashboard_tab.selectionChanged.connect(self._on_dashboard_selection_changed)
        self.dashboard_tab.view_route_btn.clicked.connect(self._view_route_on_map)
        self.tabs.addTab(self.dashboard_tab, "Live Tracking")

        # Tab 2: Summary Comparison
        self.summary_tab = SummaryStatisticsTab(self.policy_names)
        self.summary_tab.selectionChanged.connect(self._on_summary_selection_changed)
        self.summary_tab.redrawRequested.connect(self.redraw_summary_chart)
        self.tabs.addTab(self.summary_tab, "Summary Analysis")

        # Tab 3: Raw Logs
        self.raw_log_area = QTextEdit()
        self.raw_log_area.setReadOnly(True)
        self.tabs.addTab(self.raw_log_area, "Raw Output")

    def setup_workers(self, log_path):
        """Initialize and start background workers."""
        # File Tailer
        self.file_thread = QThread()
        self.file_tailer = FileTailerWorker(data_mutex=self.data_mutex, log_path=log_path)
        self.file_tailer.moveToThread(self.file_thread)
        self.file_thread.started.connect(self.file_tailer.tail_file)
        self.file_tailer.log_line_ready.connect(self._handle_new_log_line)
        self.file_thread.start()

        # Chart Processor (Legacy support for ChartWorker)
        self.chart_thread = QThread()
        # Note: ChartWorker expects specific dict structures, we bridge them here
        self.chart_worker = ChartWorker(
            daily_data=self.data_manager.accumulated_data,  # Compatibility bridge
            metrics_to_plot=TARGET_METRICS,
            data_mutex=self.data_mutex,
            historical_bin_data=self.data_manager.day_data,  # Simplified bridge
            latest_bin_data={},
        )
        self.chart_worker.moveToThread(self.chart_thread)
        self.chart_thread.start()

        self.start_chart_processing.connect(self.chart_worker.process_data)
        self.chart_worker.data_ready.connect(self._update_ui_on_data_ready)

    @Slot(str)
    def _handle_new_log_line(self, line):
        """Process incoming log lines and update components."""
        self.raw_log_area.append(line.strip())

        if "GUI_DAY_LOG_START:" in line:
            record = self.data_manager.parse_log_line(line.split("GUI_DAY_LOG_START:")[1])
            if record:
                key, policy, sample = self.data_manager.process_record(record)
                if key:
                    self.status_label.setText(f"Processing: {policy} - Sample {sample}")
                    self.dashboard_tab.update_samples(list(self.data_manager.policy_samples[policy]))  # type: ignore[index]
                    self.dashboard_tab.update_metrics(list(self.data_manager.metrics))
                    self._update_route_cache(key, record)
                    self.start_chart_processing.emit(key)

        elif "GUI_SUMMARY_LOG_START:" in line:
            self.status_label.setText("Simulation Complete.")
            self.redraw_summary_chart()

    def _update_route_cache(self, key, record):
        """
        Update the internal cache of routes for map visualization.

        Args:
            key (str): The policy/sample identifier.
            record (dict): Parsed log data containing route information.
        """
        if "routes" in record:
            if key not in self.historical_routes:
                self.historical_routes[key] = {}
            self.historical_routes[key][record.get("day", 0)] = {
                "routes": record["routes"],
                "tour_indices": record.get("tour_indices"),
                "mandatory_nodes": record.get("mandatory_nodes"),
                "bin_state_c": record.get("bin_state_c"),
                "bin_state_collected": record.get("bin_state_collected"),
            }

    @Slot(str, dict)
    def _update_ui_on_data_ready(self, target_key, processed_data):
        """
        Update the UI components when new processed simulation data is available.

        Args:
            target_key (str): Identifier for the policy/sample being updated.
            processed_data (dict): The data to be plotted.
        """
        # delegation to dashboard_tab or summary_tab based on current key
        # For brevity, assume integration with Matplotlib canvases in the tabs
        self.dashboard_tab.day_combo.clear()
        days = sorted(self.data_manager.day_data.get(target_key, {}).keys())
        self.dashboard_tab.day_combo.addItems([str(d) for d in days])
        self.dashboard_tab.view_route_btn.setEnabled(len(days) > 0)

    def _on_dashboard_selection_changed(self):
        """Handle control changes in the dashboard."""
        policy = self.dashboard_tab.policy_combo.currentText()
        sample = self.dashboard_tab.sample_combo.currentText()
        if policy and sample:
            self.start_chart_processing.emit(f"{policy}_{sample}")

    def _on_summary_selection_changed(self):
        """Handle control changes in the summary tab."""
        pass

    def redraw_summary_chart(self):
        """Update summary visualization."""
        pass

    def _view_route_on_map(self):
        """
        Generate a Folium-based HTML map for the selected route and open it in the web browser.

        Uses the loaded dataset for bin coordinates and the per-day cache
        for tour indices, mandatory node selections, and bin states.
        """
        policy = self.dashboard_tab.policy_combo.currentText()
        sample = self.dashboard_tab.sample_combo.currentText()
        day_str = self.dashboard_tab.day_combo.currentText()
        if not (policy and sample and day_str):
            return

        key = f"{policy}_{sample}"
        day = int(day_str)
        day_cache = self.historical_routes.get(key, {}).get(day)
        if not day_cache:
            return

        routes = day_cache.get("routes", [])
        if not routes:
            return

        tour_indices_set = set(day_cache.get("tour_indices") or [])
        mandatory_nodes_set = self._normalize_mandatory_nodes(day_cache.get("mandatory_nodes") or [], routes)
        bin_states = day_cache.get("bin_state_c") or []
        collected = day_cache.get("bin_state_collected") or []

        collected_set = {i for i, amt in enumerate(collected) if amt > 0}

        # Center on first route point
        center = [routes[0].get("lat", 0), routes[0].get("lng", 0)]
        m = folium.Map(location=center, zoom_start=13, tiles="cartodbpositron")

        # Depot marker
        for point in routes:
            if point.get("type") == "depot" and "lat" in point and "lng" in point:
                folium.Marker(
                    location=[point["lat"], point["lng"]],
                    popup=f"Depot<br>Lat: {point['lat']:.4f}<br>Lng: {point['lng']:.4f}",
                    icon=folium.Icon(color="blue", icon="home", prefix="fa"),
                ).add_to(m)
                break

        # All bins from dataset (toured + non-toured)
        sample_idx = int(sample)
        if self.dataset is not None and sample_idx < len(self.dataset):
            self._add_bin_markers_to_map(
                m,
                self.dataset[sample_idx],
                routes,
                tour_indices_set,
                collected_set,
                mandatory_nodes_set,
                bin_states,
            )

        # Route polyline
        route_coords = [(p["lat"], p["lng"]) for p in routes if "lat" in p and "lng" in p]
        if len(route_coords) > 1:
            folium.PolyLine(route_coords, color="#3388ff", weight=3, opacity=0.8).add_to(m)

        temp_path = os.path.join(udef.ROOT_DIR, "temp", f"route_{key}_{day}.html")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        m.save(temp_path)
        webbrowser.open(f"file://{os.path.abspath(temp_path)}")

    @staticmethod
    def _normalize_mandatory_nodes(mandatory_nodes_raw, routes):
        """Normalize mandatory node IDs to positional indices using the route coord mapping.

        Logs may contain either positional indices or dataset IDs depending on when
        they were generated. This method adds both representations to the set so
        downstream matching works for either format.
        """
        ds_id_to_pos = {}
        for rpt in routes:
            pos_id = rpt.get("id")
            ds_id = rpt.get("dataset_id")
            if pos_id is not None and ds_id is not None and int(pos_id) >= 0:
                ds_id_to_pos[int(ds_id)] = int(pos_id)

        result = set()
        for mg in mandatory_nodes_raw:
            ms_int = int(mg)
            result.add(ms_int)
            if ms_int in ds_id_to_pos:
                result.add(ds_id_to_pos[ms_int])
        return result

    @staticmethod
    def _add_bin_markers_to_map(
        m, sample_data, routes, tour_indices_set, collected_set, mandatory_nodes_set, bin_states
    ):
        """Add bin CircleMarkers to the Folium map with appropriate styling."""
        locs = sample_data["locs"]
        depot = sample_data["depot"]

        # If no depot marker was found in routes, add from dataset
        if not any(p.get("type") == "depot" for p in routes):
            folium.Marker(
                location=[float(depot[0]), float(depot[1])],
                popup=f"Depot<br>Lat: {depot[0]:.4f}<br>Lng: {depot[1]:.4f}",
                icon=folium.Icon(color="blue", icon="home", prefix="fa"),
            ).add_to(m)

        for bin_id in range(len(locs)):
            lat, lng = float(locs[bin_id, 0]), float(locs[bin_id, 1])

            is_toured = bin_id in tour_indices_set
            is_served = bin_id in collected_set
            is_mandatory = bin_id in mandatory_nodes_set

            fill = bin_states[bin_id] if 0 <= bin_id < len(bin_states) else 50.0
            color = "#fd7e14" if is_mandatory else "#28a745" if is_served else "#dc3545"
            radius = (5 + (fill / 100.0) * 10) if is_toured else (4 + (fill / 100.0) * 4)
            opacity = 0.7 if is_toured else 0.35
            weight = 4 if is_mandatory else 2

            popup = f"Bin {bin_id}<br>Lat: {lat:.4f}<br>Lng: {lng:.4f}<br>Fill: {fill:.1f}%"
            if is_mandatory:
                popup += "<br><b style='color: #fd7e14;'>Mandatory</b>"
            if is_served:
                popup += "<br><b style='color: #28a745;'>Served</b>"
            if not is_toured:
                popup += "<br><i>Not in route</i>"

            folium.CircleMarker(
                location=[lat, lng],
                radius=radius,
                popup=popup,
                color=color,
                fill=True,
                fillColor="#fd7e14" if is_mandatory and is_served else color,
                fillOpacity=opacity,
                weight=weight,
            ).add_to(m)

    def closeEvent(self, event):
        """Cleanup threads on close."""
        self.file_tailer.stop()
        self.file_thread.quit()
        self.file_thread.wait()
        self.chart_thread.quit()
        self.chart_thread.wait()
        event.accept()
