import os
import json
import folium
import random
import tempfile
import webbrowser
import numpy as np
import src.utils.definitions as udef

from collections import defaultdict
from PySide6.QtWidgets import (
    QSizePolicy, QComboBox, QPushButton,
    QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QTextEdit, 
)
from PySide6.QtCore import (
    Qt, Signal, QThread, 
    Slot, QMutex, QMutexLocker,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from ..helpers import ChartWorker, FileTailerWorker
from ..app_definitions import TARGET_METRICS, HEATMAP_METRICS


class SimulationResultsWindow(QWidget):
    start_chart_processing = Signal(str) 

    def __init__(self, policy_names, log_path=None):
        super().__init__()
        self.setWindowTitle("Simulation Chart and Raw Output")
        self.setWindowFlags(self.windowFlags() | Qt.Window) 

        self.data_mutex = QMutex() 
        
        # Stores scalar time-series data
        self.daily_data = defaultdict(lambda: defaultdict(dict))
        
        # Stores the LATEST snapshot for Live Bar Charts
        self.latest_bin_data = defaultdict(dict) 
        
        # Stores FULL history for Heatmaps
        # Structure: historical_bin_data[policy_sample_key][metric][day] = [array]
        self.historical_bin_data = defaultdict(lambda: defaultdict(dict))

        # Stores route coordinates for Folium visualization [NEW]
        # Structure: historical_routes[policy_sample_key][day] = [ {lat, lng, popup, type}, ... ]
        self.historical_routes = defaultdict(dict)
        
        # Tracks which samples are available for which policy for the dropdowns
        # Structure: { 'policy_name': {0, 1, 2} }
        self.available_samples_dict = defaultdict(set)

        self.policy_names = policy_names
        self.active_sample_keys = set() 
        self.summary_data = {} 
        self.policy_chart_widgets = {} 
        
        main_layout = QVBoxLayout(self)
        self.status_label = QLabel("Waiting for simulation to start...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(self.status_label)
        
        self.tabs = QTabWidget()
        # Connect tab change to refresh dropdowns if Summary is selected
        self.tabs.currentChanged.connect(self._on_main_tab_changed)
        main_layout.addWidget(self.tabs)
        
        self.policy_tabs_container = QTabWidget() 
        self.tabs.addTab(self.policy_tabs_container, "Live Tracking (Per Sample)")
        
        self.setup_raw_log_area()
        self.setup_summary_area()

        # --- FILE TAILER WORKER SETUP [NEW] ---
        # 1. Initialize Thread and Worker
        self.file_thread = QThread()
        self.file_tailer = FileTailerWorker(data_mutex=self.data_mutex, log_path=log_path) # Use the same mutex
        self.file_tailer.moveToThread(self.file_thread)
        
        # 2. Connect Signals
        self.file_thread.started.connect(self.file_tailer.tail_file)
        self.file_tailer.log_line_ready.connect(self._process_single_line_on_main_thread)
        
        # 3. Start the Thread
        self.file_thread.start()

        self.plot_color = "#3465a4" 
        
        # --- THREAD SETUP ---
        self.chart_thread = QThread()
        self.chart_worker = ChartWorker(
            daily_data=self.daily_data, 
            metrics_to_plot=TARGET_METRICS,
            data_mutex=self.data_mutex,
            historical_bin_data = self.historical_bin_data,
            latest_bin_data = self.latest_bin_data
        )
        self.chart_worker.moveToThread(self.chart_thread)
        self.chart_thread.start()
        
        self.start_chart_processing.connect(self.chart_worker.process_data)
        self.chart_worker.data_ready.connect(self._update_chart_on_main_thread) 

    # -------------------------------------------------------------------------
    # LIVE UPDATES (Main Thread)
    # -------------------------------------------------------------------------
    @Slot(str, dict)
    def _update_chart_on_main_thread(self, target_key, processed_data):
        if target_key not in self.policy_chart_widgets:
            return

        widgets = self.policy_chart_widgets[target_key]

        # 1. Update Line Charts (Time Series)
        line_axes = widgets['line_axes']
        line_canvas = widgets['line_canvas']
        max_days = processed_data['max_days']

        for i, metric in enumerate(TARGET_METRICS):
            ax = line_axes[i]
            ax.clear() 
            ax.grid(True)
            ax.set_title(f'{metric}')
            ax.set_ylabel(metric)

            metric_data = processed_data['metrics'].get(metric)
            if metric_data and metric_data['days']:
                ax.plot(metric_data['days'], metric_data['values'], 
                        marker='o', markersize=3, linestyle='-', color=self.plot_color)

                # Dynamic scaling
                vals = metric_data['values']
                if vals:
                    min_v, max_v = min(vals), max(vals)
                    rng = max_v - min_v
                    buffer = max(1.0, rng * 0.1) if rng > 0 else max(1.0, abs(max_v * 0.1))
                    ax.set_ylim(min_v - buffer, max_v + buffer)
                else:
                    ax.set_ylim(-0.1, 1.1)

            ax.set_xlim(0, max(10, max_days + 1))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if i == len(TARGET_METRICS) - 1: ax.set_xlabel("Day")

        line_canvas.draw_idle()

        # 2. Update Live Bar Charts Dropdown & Plot
        self._update_day_dropdown_and_bars(target_key)

    def _update_day_dropdown_and_bars(self, target_key):
        """Updates the dropdown with available days and triggers the bar plot."""
        if target_key not in self.historical_bin_data: return
        
        widgets = self.policy_chart_widgets[target_key]
        combo = widgets['day_selector']
        view_route_btn = widgets['view_route_btn']
        
        # Get list of days recorded so far (based on 'bin_state_c')
        hist_data = self.historical_bin_data[target_key].get('bin_state_c', {})
        if not hist_data: 
            view_route_btn.setEnabled(False)
            return
        
        available_days = sorted(hist_data.keys())
        current_count = combo.count()
        
        # If we have new days, add them to the dropdown
        if len(available_days) > current_count:
            combo.blockSignals(True) 
            
            was_at_latest = (combo.currentIndex() == current_count - 1)
            
            combo.clear()
            combo.addItems([str(d) for d in available_days])
            
            if was_at_latest or current_count == 0:
                combo.setCurrentIndex(combo.count() - 1)
            
            combo.blockSignals(False)
            
            if was_at_latest or current_count == 0:
                self._draw_bars_for_selected_day(target_key)
        
        # Enable route button if we have data for current day
        self._update_route_button_state(target_key)

    def _update_route_button_state(self, target_key):
        widgets = self.policy_chart_widgets[target_key]
        combo = widgets['day_selector']
        btn = widgets['view_route_btn']
        
        if combo.count() == 0:
            btn.setEnabled(False)
            return
            
        try:
            day = int(combo.currentText())
            has_route = day in self.historical_routes.get(target_key, {})
            btn.setEnabled(has_route)
        except ValueError:
            btn.setEnabled(False)

    def _draw_bars_for_selected_day(self, target_key):
        """Draws Bar Charts for the day selected in the QComboBox."""
        widgets = self.policy_chart_widgets[target_key]
        combo = widgets['day_selector']
        
        selected_day_str = combo.currentText()
        if not selected_day_str: return
        
        selected_day = int(selected_day_str)
        
        bar_axes = widgets['hm_axes']
        bar_canvas = widgets['hm_canvas']
        info_label = widgets['hm_info_label']
        
        # Retrieve data from HISTORY based on selected day
        hist = self.historical_bin_data[target_key]
        
        info_label.setText(f"<b>Viewing Day:</b> {selected_day}")
        
        # Update route button availability for this new day
        self._update_route_button_state(target_key)

        titles = {
            'bin_state_c': "Fill Level (%)",
            'bin_state_collected': "Waste Collected (kg)",
            'bins_state_c_after': "Fill Level After Collection (%)"
        }
        
        for i, metric in enumerate(HEATMAP_METRICS):
            ax = bar_axes[i]
            ax.clear()
            
            day_data = hist.get(metric, {}).get(selected_day, [])
            
            if day_data:
                x_indices = np.arange(len(day_data))
                if metric in ['bin_state_c', 'bins_state_c_after']:
                    ax.bar(x_indices, day_data, color='#e67e22', edgecolor='black') 
                    ax.set_ylim(0, 105)
                    ax.axhline(y=100, color='red', linestyle='--', linewidth=1, label="Overflow")
                else:
                    ax.bar(x_indices, day_data, color='#2980b9', edgecolor='black')
                
                ax.set_ylabel("Value")
                ax.set_title(titles.get(metric, metric))
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)

                # Set X-limits from -0.5 to N - 0.5 to center the 0 and N-1 bars correctly
                ax.set_xlim(-0.5, len(day_data) - 0.5) 
                
                # Set X-ticks for every bin index
                ax.set_xticks(x_indices)
            else:
                ax.text(0.5, 0.5, "No Data for Day", ha='center', va='center')
            
            if i == len(HEATMAP_METRICS) - 1:
                ax.set_xlabel("Bin Index")
        
        bar_canvas.draw_idle()
    
    def _view_route_for_selected_day(self, target_key):
        """Generates and opens a Folium map for the route of the selected day."""
        widgets = self.policy_chart_widgets[target_key]
        combo = widgets['day_selector']
        selected_day_str = combo.currentText()
        if not selected_day_str: return
        
        day = int(selected_day_str)
        route_points = self.historical_routes.get(target_key, {}).get(day, [])
        
        if not route_points:
            print(f"No route data available for {target_key} on day {day}")
            return
            
        # Extract valid points (those with lat/lng)
        points_to_plot = []
        depot_loc = None
        
        for p in route_points:
            if 'lat' in p and 'lng' in p:
                points_to_plot.append(p)
                if p.get('type') == 'depot':
                    depot_loc = (p['lat'], p['lng'])
        
        if not points_to_plot:
            print("Route data exists but contains no valid coordinates.")
            return

        # Determine map center
        if depot_loc:
            map_center = depot_loc
        else:
            map_center = (points_to_plot[0]['lat'], points_to_plot[0]['lng'])

        m = folium.Map(location=map_center, zoom_start=13)

        # Plot points and lines
        poly_coords = []
        
        for p in points_to_plot:
            lat, lng = p['lat'], p['lng']
            poly_coords.append((lat, lng))
            
            if p.get('type') == 'depot':
                folium.Marker(
                    location=(lat, lng),
                    popup=p.get('popup', "Depot"),
                    icon=folium.Icon(color='green', icon='home')
                ).add_to(m)
            else:
                folium.CircleMarker(
                    location=(lat, lng),
                    radius=6,
                    color='gray',
                    fill=True,
                    fill_opacity=0.7,
                    popup=p.get('popup', "Bin")
                ).add_to(m)
        
        # Draw the route line
        if len(poly_coords) > 1:
            folium.PolyLine(poly_coords, color='blue', weight=2, opacity=0.8).add_to(m)

        # Save to temp file and open
        try:
            # --- MODIFIED FILE HANDLING: Use ROOT_DIR/temp ---
            # 1. Define the custom temp directory path requested by the user
            temp_root = os.path.join(udef.ROOT_DIR, 'temp')
            os.makedirs(temp_root, exist_ok=True)
            
            # Create a unique but predictable file path
            temp_filename = f"route_{target_key.replace(' ', '_')}_{day}.html"
            temp_path = os.path.join(temp_root, temp_filename)
            
            # 2. Save the Folium map to the file path
            m.save(temp_path)
            
            # 3. Open the file in the web browser using the absolute path for reliability
            webbrowser.open('file://' + os.path.abspath(temp_path))
            
            print(f"Map saved and opened from: {os.path.abspath(temp_path)}")

        except Exception as e:
            print(f"Failed to open map: {e}")


    # -------------------------------------------------------------------------
    # SUMMARY & HEATMAPS (With Dropdowns)
    # -------------------------------------------------------------------------
    def _on_main_tab_changed(self, index):
        """Called when user switches tabs. Used to refresh summary dropdowns."""
        # Assuming "Average and StdDev (Summary)" is the last tab or identified by name
        if self.tabs.tabText(index) == "Average and StdDev (Summary)":
            self._populate_summary_policy_combo()

    def _populate_summary_policy_combo(self):
        """Populates the Policy dropdown in the Summary tab."""
        self.summary_policy_combo.blockSignals(True)
        self.summary_policy_combo.clear()
        
        # Use keys from available_samples_dict which is populated during parsing
        policies = sorted(self.available_samples_dict.keys())
        self.summary_policy_combo.addItems(policies)
        
        self.summary_policy_combo.blockSignals(False)
        
        # Trigger sample population for the first item
        if policies:
            self._populate_summary_sample_combo()

    def _populate_summary_sample_combo(self):
        """Populates the Sample dropdown based on selected Policy."""
        policy = self.summary_policy_combo.currentText()
        if not policy: return

        self.summary_sample_combo.blockSignals(True)
        self.summary_sample_combo.clear()
        
        samples = sorted(list(self.available_samples_dict.get(policy, [])))
        self.summary_sample_combo.addItems([str(s) for s in samples])
        
        self.summary_sample_combo.blockSignals(False)
        
        # Draw immediately if data exists
        if samples:
            self._draw_selected_summary_heatmap()

    def _draw_selected_summary_heatmap(self):
        """Draws the heatmap for the specific Policy + Sample selected."""
        policy = self.summary_policy_combo.currentText()
        sample_str = self.summary_sample_combo.currentText()
        
        if not policy or not sample_str: return
        
        # Reconstruct the key used in storage
        target_key = f"{policy} sample {sample_str}"
        
        self.hm_summary_fig.clear()
        
        # If data doesn't exist yet (or mutex locked elsewhere), handle gracefully
        if target_key not in self.historical_bin_data:
            ax = self.hm_summary_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Loading Data...", ha='center', va='center')
            self.hm_summary_canvas.draw_idle()
            return

        with QMutexLocker(self.data_mutex):
            hist_data = self.historical_bin_data[target_key]
            
            # Create subplots
            axes = self.hm_summary_fig.subplots(len(HEATMAP_METRICS), 1, sharex=True)
            if len(HEATMAP_METRICS) == 1: axes = [axes]
            
            self.hm_summary_fig.tight_layout(pad=3.0)
            self.hm_summary_fig.subplots_adjust(top=0.92, bottom=0.1)

            titles = {
                'bin_state_c': "Fill Level (%) - Fire Scale",
                'bin_state_collected': "Waste Collected (kg)",
                'bins_state_c_after': "Fill Level After Collection (%)"
            }

            for i, metric in enumerate(HEATMAP_METRICS):
                ax = axes[i]
                metric_history = hist_data.get(metric, {})
                
                if not metric_history:
                    ax.text(0.5, 0.5, "No Data for Metric", ha='center', va='center')
                    continue

                days = sorted(metric_history.keys())
                # Create matrix (Days x Bins)
                mat = np.array([metric_history[d] for d in days])
                
                # Plot
                cmap = 'viridis'
                vmin, vmax = None, None
                
                if metric in ['bin_state_c', 'bins_state_c_after']:
                    cmap = 'hot' # Fire color
                    vmin, vmax = 30, 70 # Contrast
                
                im = ax.imshow(mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
                
                ax.set_title(titles.get(metric, metric))
                ax.set_ylabel("Day")
                
                # Optional: Add day ticks if not too crowded
                if len(days) < 20:
                    ax.set_yticks(np.arange(len(days)))
                    ax.set_yticklabels(days)

            axes[-1].set_xlabel("Bin Index")
            
        self.hm_summary_canvas.draw_idle()

    def redraw_summary_chart(self):
        """Updates the Bar Chart based on dropdown selection ('All Metrics' or specific)."""
        if not self.summary_data: return

        self.summary_ax.clear()
        
        log = self.summary_data['log']
        log_std = self.summary_data['log_std']
        policy_names = self.summary_data['policies']
        n_policies = len(policy_names)
        colors = self._generate_distinct_colors(n_policies)
        
        selection = self.summary_metric_combo.currentText()

        # ---------------------------------------------------------
        # MODE A: View All Metrics (Grouped Bar Chart)
        # ---------------------------------------------------------
        if selection == "All Metrics":
            n_metrics = len(udef.SIM_METRICS)
            bar_width = 0.8 / n_policies
            x = np.arange(n_metrics)

            for i, policy in enumerate(policy_names):
                means = [log[policy][j] for j in range(n_metrics)]
                stds = [log_std[policy][j] for j in range(n_metrics)]
                
                # Offset bars based on policy index
                r = x + bar_width * i
                
                self.summary_ax.bar(r, means, width=bar_width, 
                                    edgecolor='grey', label=policy,
                                    yerr=stds, capsize=5, color=colors[i % len(colors)])

            self.summary_ax.set_ylabel("Mean Value")
            self.summary_ax.set_title("Metrics Average (Across Samples)")
            # Center ticks among the groups
            self.summary_ax.set_xticks(x + bar_width * (n_policies - 1) / 2, udef.SIM_METRICS)
            self.summary_ax.tick_params(axis='x', rotation=45)
            self.summary_ax.legend()

        # ---------------------------------------------------------
        # MODE B: View Single Metric (Comparison Across Policies)
        # ---------------------------------------------------------
        else:
            if selection not in udef.SIM_METRICS: return
            metric_idx = udef.SIM_METRICS.index(selection)
            
            means = []
            stds = []
            
            for policy in policy_names:
                means.append(log[policy][metric_idx])
                stds.append(log_std[policy][metric_idx])
            
            x_indices = np.arange(len(policy_names))
            
            # Draw bars (one per policy)
            self.summary_ax.bar(x_indices, means, yerr=stds, 
                                align='center', alpha=0.8, 
                                edgecolor='black', capsize=10, 
                                color=colors) # Use distinct colors for each policy

            self.summary_ax.set_ylabel(f"Mean {selection}")
            self.summary_ax.set_title(f"Comparison: {selection}")
            self.summary_ax.set_xticks(x_indices)
            self.summary_ax.set_xticklabels(policy_names, rotation=45, ha='right')

        # Shared Formatting
        self.summary_ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        self.summary_fig.tight_layout()
        self.summary_canvas.draw_idle()
        
        # Refresh dropdowns in Heatmap Tab if needed
        if self.summary_policy_combo.count() == 0:
            self._populate_summary_policy_combo()

    # -------------------------------------------------------------------------
    # SETUP & PARSING
    # -------------------------------------------------------------------------
    def setup_summary_area(self):
        self.summary_tab = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_tab)
        
        self.summary_nested_tabs = QTabWidget()
        self.summary_layout.addWidget(self.summary_nested_tabs)

        # --- Tab 1: Scalar Summary ---
        self.scalar_summary_widget = QWidget()
        scalar_layout = QVBoxLayout(self.scalar_summary_widget)
        
        # [MODIFIED] Dropdown setup
        control_layout = QHBoxLayout()
        lbl_metric = QLabel("Select Metric:")
        lbl_metric.setStyleSheet("font-weight: bold;")
        
        self.summary_metric_combo = QComboBox()
        self.summary_metric_combo.setMinimumWidth(200)
        
        # Add "All Metrics" first, then the specific options
        self.summary_metric_combo.addItem("All Metrics")
        self.summary_metric_combo.addItems(udef.SIM_METRICS)
        
        self.summary_metric_combo.currentTextChanged.connect(self.redraw_summary_chart)
        
        control_layout.addWidget(lbl_metric)
        control_layout.addWidget(self.summary_metric_combo)
        control_layout.addStretch()
        
        scalar_layout.addLayout(control_layout)

        # Chart Canvas
        self.summary_fig = Figure(figsize=(10, 6))
        self.summary_canvas = FigureCanvas(self.summary_fig)
        scalar_layout.addWidget(self.summary_canvas)
        self.summary_ax = self.summary_fig.add_subplot(111)
        
        self.summary_nested_tabs.addTab(self.scalar_summary_widget, "Metrics (Bar Chart)")

        # --- Tab 2: Heatmaps (Keep as is) ---
        self.hm_summary_widget = QWidget()
        # ... (Rest of existing heatmap setup code remains unchanged) ...
        hm_layout = QVBoxLayout(self.hm_summary_widget)
        
        controls_layout = QHBoxLayout()
        lbl_pol = QLabel("Policy:")
        self.summary_policy_combo = QComboBox()
        self.summary_policy_combo.setMinimumWidth(150)
        self.summary_policy_combo.currentIndexChanged.connect(self._populate_summary_sample_combo)
        
        lbl_sam = QLabel("Sample:")
        self.summary_sample_combo = QComboBox()
        self.summary_sample_combo.setMinimumWidth(100)
        self.summary_sample_combo.currentIndexChanged.connect(self._draw_selected_summary_heatmap)
        
        controls_layout.addWidget(lbl_pol)
        controls_layout.addWidget(self.summary_policy_combo)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(lbl_sam)
        controls_layout.addWidget(self.summary_sample_combo)
        controls_layout.addStretch()
        
        hm_layout.addLayout(controls_layout)

        self.hm_summary_fig = Figure(figsize=(10, 8)) 
        self.hm_summary_canvas = FigureCanvas(self.hm_summary_fig)
        self.hm_summary_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        hm_layout.addWidget(self.hm_summary_canvas)

        self.summary_nested_tabs.addTab(self.hm_summary_widget, "Fill History (Heatmaps)")

        self.tabs.addTab(self.summary_tab, "Average and StdDev (Summary)")

    def add_sample_chart_tab(self, policy_sample_key):
        """Creates tab with Time Series and Day-Selectable Bar Charts."""
        main_container = QWidget()
        main_layout = QVBoxLayout(main_container)
        sub_tabs = QTabWidget()
        main_layout.addWidget(sub_tabs)
        
        # Sub-Tab 1: Line Charts
        line_tab = QWidget(); line_layout = QVBoxLayout(line_tab)
        line_fig = Figure(figsize=(10, 6))
        line_canvas = FigureCanvas(line_fig)
        line_axes = line_fig.subplots(len(TARGET_METRICS), 1, sharex=True)
        line_fig.tight_layout(pad=2.0)
        line_layout.addWidget(line_canvas)
        sub_tabs.addTab(line_tab, "Evolution (Time Series)")

        # Sub-Tab 2: Bin State with Dropdown
        hm_tab = QWidget(); hm_layout = QVBoxLayout(hm_tab)
        
        control_layout = QHBoxLayout()
        day_label = QLabel("Select Day:")
        day_selector = QComboBox()
        day_selector.setMinimumWidth(100)
        
        # -- VIEW ROUTE BUTTON [NEW] --
        view_route_btn = QPushButton("View Route")
        view_route_btn.setEnabled(False) # Disabled until data arrives
        view_route_btn.clicked.connect(
            lambda: self._view_route_for_selected_day(policy_sample_key)
        )
        # -----------------------------
        
        day_selector.currentIndexChanged.connect(
            lambda: self._draw_bars_for_selected_day(policy_sample_key)
        )
        hm_info_label = QLabel("Waiting for data...")
        
        control_layout.addWidget(day_label)
        control_layout.addWidget(day_selector)
        control_layout.addSpacing(10)
        control_layout.addWidget(view_route_btn) # Added button to layout
        control_layout.addSpacing(20)
        control_layout.addWidget(hm_info_label)
        control_layout.addStretch()
        hm_layout.addLayout(control_layout)

        hm_fig = Figure(figsize=(10, 6))
        hm_canvas = FigureCanvas(hm_fig)
        hm_canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        hm_axes = hm_fig.subplots(len(HEATMAP_METRICS), 1, sharex=True)
        hm_fig.tight_layout(pad=3.0)
        hm_layout.addWidget(hm_canvas)
        sub_tabs.addTab(hm_tab, "Bin State (Bars)")

        self.policy_chart_widgets[policy_sample_key] = {
            'line_fig': line_fig, 'line_canvas': line_canvas, 'line_axes': line_axes,
            'hm_fig': hm_fig, 'hm_canvas': hm_canvas, 'hm_axes': hm_axes, 
            'hm_info_label': hm_info_label, 'day_selector': day_selector,
            'view_route_btn': view_route_btn # Stored ref
        }

        self.policy_tabs_container.addTab(main_container, policy_sample_key)
        self.active_sample_keys.add(policy_sample_key)

    def stop_thread(self):
        print("Stopping chart worker thread...")
        self.chart_thread.quit()
        self.chart_thread.wait()

    def closeEvent(self, event):
        self.stop_thread()
        event.accept()

    def stop_thread(self):
        """Safely stops all worker threads."""
        # [NEW] Stop the file tailer thread
        print("Stopping file tailer thread...")
        self.file_tailer.stop()
        self.file_thread.quit()
        self.file_thread.wait()
        
        # [Existing] Stop the chart worker thread
        print("Stopping chart worker thread...")
        self.chart_thread.quit()
        self.chart_thread.wait()

    @Slot(str)
    def _process_single_line_on_main_thread(self, line):
        """
        Slot executed on the MAIN thread. Receives one complete line from the tailer
        and sends it to the core parsing logic (parse_buffer is adapted for single lines).
        """
        # --- UPDATE START: Display raw line in log window ---
        self.raw_log_area.append(line.strip())
        # --- UPDATE END ---

        # Call the existing parsing logic, adapted to handle single lines
        self.parse_buffer(line)

    def parse_buffer(self, line: str) -> str:
        """
        Adapted to process single lines from the file tailer worker.
        Instead of handling a massive buffer, we only process the current line.
        """
        # Ensure we only process lines containing the required log markers
        if 'GUI_DAY_LOG_START:' in line or 'GUI_SUMMARY_LOG_START:' in line:
            # We don't need to split by 'GUI_' multiple times; the line is already atomic.
            self._process_single_record(line.strip())
        
        # For compatibility with the old method signature (which expected a buffer 
        # and returned the remainder), we return an empty string since the file 
        # tailer handles the stream.
        return ""

    def _process_single_record(self, record):
        if record.startswith("GUI_DAY_LOG_START:"):
            try:
                end = record.rfind('}')
                clean = record[:end+1]
                
                with QMutexLocker(self.data_mutex):
                    parts = clean.split("GUI_DAY_LOG_START:")[1].strip().split(',', 3)
                    policy = parts[0].strip(); sample = int(parts[1]); day = int(parts[2])
                    key = f"{policy} sample {sample}"

                    if key not in self.active_sample_keys: 
                        self.add_sample_chart_tab(key)
                        
                    # [NEW] Track availability for dropdowns
                    self.available_samples_dict[policy].add(sample)

                    data = json.loads(parts[3])
                    
                    # Store Route Coords [NEW]
                    if 'tour' in data:
                        self.historical_routes[key][day] = data['tour']
                    
                    for m, v in data.items():
                        if m in HEATMAP_METRICS: 
                            self.historical_bin_data[key][m][day] = v 
                        elif m in ['bin_state_travel', 'bin_state_ndays']:
                            pass
                        elif m in TARGET_METRICS or m in udef.SIM_METRICS:
                            self.daily_data[key][m][day] = float(v)
                    
                    self.status_label.setText(f"Processing: {key} day {day}")
                
                self.start_chart_processing.emit(key)
            except Exception as e: print(f"Log Error: {e}")

        elif record.startswith("GUI_SUMMARY_LOG_START:"):
            try:
                end = record.rfind('}')
                clean = record[:end+1]
                json_str = clean.split("GUI_SUMMARY_LOG_START:")[1].strip()
                self.summary_data = json.loads(json_str)
                self.status_label.setText("Simulation Complete.")
                self.redraw_summary_chart()
            except Exception as e: print(f"Summary Error: {e}")

    def setup_raw_log_area(self):
        self.raw_tab = QWidget(); self.raw_layout = QVBoxLayout(self.raw_tab)
        self.tabs.addTab(self.raw_tab, "Raw Output (Log)")
        self.raw_log_area = QTextEdit()
        self.raw_log_area.setReadOnly(True)
        self.raw_layout.addWidget(self.raw_log_area)

    def _generate_distinct_colors(self, num_colors):
        return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_colors)]