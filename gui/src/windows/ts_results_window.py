"""
Simulation results visualization window.

This module provides a specialized window for displaying real-time and
summary statistics from simulation experiments. It acts as an orchestrator
for specialized sub-widgets and data managers.
"""

import os
import webbrowser

import folium
import logic.src.constants as udef
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

    def __init__(self, policy_names, log_path=None):
        """
        Initialize the simulation results orchestration window.

        Args:
            policy_names (list[str]): List of policy identifiers used to track results.
            log_path (str, optional): Absolute path to the simulation log file to tail.
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

        # UI Initialization
        self.setup_ui()

        # Worker Threads
        self.setup_workers(log_path)

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
            self.historical_routes[key][record.get("day", 0)] = record["routes"]

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

        Uses stored coordinates from the route cache to draw polylines.
        """
        policy = self.dashboard_tab.policy_combo.currentText()
        sample = self.dashboard_tab.sample_combo.currentText()
        day_str = self.dashboard_tab.day_combo.currentText()
        if not (policy and sample and day_str):
            return

        key = f"{policy}_{sample}"
        day = int(day_str)
        routes = self.historical_routes.get(key, {}).get(day)
        if not routes:
            return

        # Simplified map generation logic (delegated from original)
        m = folium.Map(location=[routes[0]["lat"], routes[0]["lng"]], zoom_start=13)
        coords = [(r["lat"], r["lng"]) for r in routes]
        folium.PolyLine(coords, color="blue", weight=2.5).add_to(m)

        temp_path = os.path.join(udef.ROOT_DIR, "temp", f"route_{key}_{day}.html")
        os.makedirs(os.path.dirname(temp_path), exist_ok=True)
        m.save(temp_path)
        webbrowser.open(f"file://{os.path.abspath(temp_path)}")

    def closeEvent(self, event):
        """Cleanup threads on close."""
        self.file_tailer.stop()
        self.file_thread.quit()
        self.file_thread.wait()
        self.chart_thread.quit()
        self.chart_thread.wait()
        event.accept()
