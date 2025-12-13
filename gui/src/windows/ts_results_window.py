import os
import json
import folium
import random
import webbrowser
import numpy as np
import logic.src.utils.definitions as udef

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
from ..utils.app_definitions import TARGET_METRICS, HEATMAP_METRICS


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

        # Stores route coordinates for Folium visualization
        self.historical_routes = defaultdict(dict)
        
        # Tracks which samples are available for which policy
        self.available_samples_dict = defaultdict(set)

        self.policy_names = policy_names
        
        # [MODIFIED] No longer tracking active tabs, we use a single dashboard
        self.live_ui_components = {} 
        # Stores aggregated summary data from potentially multiple log lines
        self.summary_data = {
            'policies': [],
            'log': {},
            'log_std': {},
            'n_samples': 0
        }
        
        main_layout = QVBoxLayout(self)
        self.status_label = QLabel("Waiting for simulation to start...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(self.status_label)
        
        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_main_tab_changed)
        main_layout.addWidget(self.tabs)
        
        # [MODIFIED] Setup the single Live Dashboard instead of a tab container
        self.setup_live_dashboard()
        
        self.setup_raw_log_area()
        self.setup_summary_area()

        # --- WORKER SETUP ---
        self.file_thread = QThread()
        self.file_tailer = FileTailerWorker(data_mutex=self.data_mutex, log_path=log_path)
        self.file_tailer.moveToThread(self.file_thread)
        self.file_thread.started.connect(self.file_tailer.tail_file)
        self.file_tailer.log_line_ready.connect(self._process_single_line_on_main_thread)
        self.file_thread.start()

        self.plot_color = "#3465a4" 
        
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
    # [NEW] LIVE DASHBOARD SETUP
    # -------------------------------------------------------------------------
    # -------------------------------------------------------------------------
    # [NEW] LIVE DASHBOARD SETUP
    # -------------------------------------------------------------------------
    def setup_live_dashboard(self):
        """Creates the single tab that dynamically displays data based on dropdown selection."""
        dashboard_widget = QWidget()
        layout = QVBoxLayout(dashboard_widget)
        
        # 1. Controls Area (Policy & Sample Selection)
        controls_layout = QHBoxLayout()
        
        lbl_pol = QLabel("Select Policy:")
        lbl_pol.setStyleSheet("font-weight: bold;")
        self.live_policy_combo = QComboBox()
        self.live_policy_combo.setMinimumWidth(200)
        self.live_policy_combo.currentIndexChanged.connect(self._on_live_policy_changed)

        lbl_sam = QLabel("Select Sample:")
        lbl_sam.setStyleSheet("font-weight: bold;")
        self.live_sample_combo = QComboBox()
        self.live_sample_combo.setMinimumWidth(100)
        self.live_sample_combo.currentIndexChanged.connect(self._on_live_sample_changed)

        controls_layout.addWidget(lbl_pol)
        controls_layout.addWidget(self.live_policy_combo)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(lbl_sam)
        controls_layout.addWidget(self.live_sample_combo)
        controls_layout.addStretch()
        
        layout.addLayout(controls_layout)
        
        # 2. Charts Tabs (Time Series & Bar Charts)
        self.live_sub_tabs = QTabWidget()
        
        # --- Sub-Tab A: Line Charts ---
        line_tab = QWidget()
        line_layout = QVBoxLayout(line_tab)
        
        line_fig = Figure(figsize=(10, 6))
        line_canvas = FigureCanvas(line_fig)
        line_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Ensure expansion
        line_canvas.updateGeometry()
        
        line_axes = line_fig.subplots(len(TARGET_METRICS), 1, sharex=True)
        line_fig.tight_layout(pad=2.0)
        
        line_layout.addWidget(line_canvas, 1) # Add with stretch=1
        self.live_sub_tabs.addTab(line_tab, "Evolution (Time Series)")

        # --- Sub-Tab B: Bar Charts (Bin State) ---
        hm_tab = QWidget()
        hm_layout = QVBoxLayout(hm_tab)
        hm_layout.setSpacing(5) # Reduce spacing between controls and chart
        
        # Controls for Day selection within the Bar Chart tab
        hm_controls = QHBoxLayout()
        day_label = QLabel("Select Day:")
        day_selector = QComboBox()
        day_selector.setMinimumWidth(100)
        day_selector.currentIndexChanged.connect(self._draw_bars_for_selected_day)
        
        view_route_btn = QPushButton("View Route")
        view_route_btn.setEnabled(False)
        view_route_btn.clicked.connect(self._view_route_for_current_selection)

        hm_info_label = QLabel("Waiting for data...")

        hm_controls.addWidget(day_label)
        hm_controls.addWidget(day_selector)
        hm_controls.addSpacing(10)
        hm_controls.addWidget(view_route_btn)
        hm_controls.addSpacing(20)
        hm_controls.addWidget(hm_info_label)
        hm_controls.addStretch()
        
        # Add controls to layout
        hm_layout.addLayout(hm_controls)

        # Setup Chart
        hm_fig = Figure(figsize=(10, 6))
        hm_canvas = FigureCanvas(hm_fig)
        hm_canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding) # Ensure expansion
        hm_canvas.updateGeometry()
        
        hm_axes = hm_fig.subplots(len(HEATMAP_METRICS), 1, sharex=True)
        hm_fig.tight_layout(pad=3.0)
        
        # Add Chart to layout with Stretch Factor 1 (Consumes all remaining vertical space)
        hm_layout.addWidget(hm_canvas, 1)
        
        self.live_sub_tabs.addTab(hm_tab, "Bin State (Bars)")
        layout.addWidget(self.live_sub_tabs)

        # 3. Store References
        self.live_ui_components = {
            'line_fig': line_fig, 
            'line_canvas': line_canvas, 
            'line_axes': line_axes,
            'hm_fig': hm_fig, 
            'hm_canvas': hm_canvas, 
            'hm_axes': hm_axes, 
            'hm_info_label': hm_info_label, 
            'day_selector': day_selector,
            'view_route_btn': view_route_btn
        }

        self.tabs.addTab(dashboard_widget, "Live Tracking")

    # -------------------------------------------------------------------------
    # [NEW] DROPDOWN LOGIC
    # -------------------------------------------------------------------------
    def _update_live_dropdowns(self, policy, sample):
        """Called when new data arrives. Updates dropdowns without breaking state."""
        # 1. Update Policy List
        if self.live_policy_combo.findText(policy) == -1:
            self.live_policy_combo.addItem(policy)

        # 2. Update Sample List (only if currently selected policy matches, or empty)
        current_policy = self.live_policy_combo.currentText()
        if not current_policy or current_policy == policy:
            sample_str = str(sample)
            if self.live_sample_combo.findText(sample_str) == -1:
                self.live_sample_combo.addItem(sample_str)

    def _on_live_policy_changed(self):
        """Refresh sample list when policy changes."""
        policy = self.live_policy_combo.currentText()
        if not policy: return

        self.live_sample_combo.blockSignals(True)
        self.live_sample_combo.clear()
        
        samples = sorted(list(self.available_samples_dict.get(policy, [])))
        self.live_sample_combo.addItems([str(s) for s in samples])
        
        self.live_sample_combo.blockSignals(False)
        
        # Trigger chart refresh for the new default sample (index 0)
        self._on_live_sample_changed()

    def _on_live_sample_changed(self):
        """Trigger a manual chart refresh when the user selects a different sample."""
        target_key = self._get_current_key()
        if target_key:
            # Request the worker to process/re-emit data for this key
            self.start_chart_processing.emit(target_key)
            # Also manually update the day dropdown
            self._update_day_dropdown(target_key)

    def _get_current_key(self):
        p = self.live_policy_combo.currentText()
        s = self.live_sample_combo.currentText()
        if p and s:
            return f"{p} sample {s}"
        return None

    # -------------------------------------------------------------------------
    # LIVE UPDATES
    # -------------------------------------------------------------------------
    @Slot(str, dict)
    def _update_chart_on_main_thread(self, target_key, processed_data):
        # [MODIFIED] Only update the charts if the processed data matches the user's selection
        current_key = self._get_current_key()
        if target_key != current_key:
            return

        widgets = self.live_ui_components
        line_axes = widgets['line_axes']
        line_canvas = widgets['line_canvas']
        max_days = processed_data['max_days']

        # 1. Update Line Charts
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
                
                # Simple autoscaling
                vals = metric_data['values']
                if vals:
                    min_v, max_v = min(vals), max(vals)
                    rng = max_v - min_v
                    buffer = max(1.0, rng * 0.1) if rng > 0 else max(1.0, abs(max_v * 0.1))
                    ax.set_ylim(min_v - buffer, max_v + buffer)

            ax.set_xlim(0, max(10, max_days + 1))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            if i == len(TARGET_METRICS) - 1: ax.set_xlabel("Day")

        line_canvas.draw_idle()

        # 2. Update Day Dropdown for Bars
        self._update_day_dropdown(target_key)

    def _update_day_dropdown(self, target_key):
        widgets = self.live_ui_components
        combo = widgets['day_selector']
        
        hist_data = self.historical_bin_data[target_key].get('bin_state_c', {})
        available_days = sorted(hist_data.keys())
        
        current_count = combo.count()
        if len(available_days) > current_count:
            combo.blockSignals(True)
            was_at_latest = (combo.currentIndex() == current_count - 1)
            
            combo.clear()
            combo.addItems([str(d) for d in available_days])
            
            # Auto-select latest if user was already at latest or it's new
            if was_at_latest or current_count == 0:
                combo.setCurrentIndex(combo.count() - 1)
            
            combo.blockSignals(False)
            
            # Trigger draw if we just auto-selected
            if was_at_latest or current_count == 0:
                self._draw_bars_for_selected_day()
        
        self._update_route_button_state()

    def _draw_bars_for_selected_day(self):
        """Draws Bar Charts based on current Policy/Sample and current Day combo."""
        target_key = self._get_current_key()
        if not target_key: return

        widgets = self.live_ui_components
        combo = widgets['day_selector']
        selected_day_str = combo.currentText()
        if not selected_day_str: return
        
        selected_day = int(selected_day_str)
        
        bar_axes = widgets['hm_axes']
        bar_canvas = widgets['hm_canvas']
        info_label = widgets['hm_info_label']
        
        hist = self.historical_bin_data[target_key]
        info_label.setText(f"<b>Viewing Day:</b> {selected_day}")
        self._update_route_button_state()

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
                color = '#e67e22' if metric in ['bin_state_c', 'bins_state_c_after'] else '#2980b9'
                ax.bar(x_indices, day_data, color=color, edgecolor='black')
                
                if metric in ['bin_state_c', 'bins_state_c_after']:
                    ax.set_ylim(0, 105)
                    ax.axhline(y=100, color='red', linestyle='--', label="Overflow")
                
                ax.set_title(titles.get(metric, metric))
                ax.grid(True, axis='y', linestyle='--', alpha=0.7)
                ax.set_xlim(-0.5, len(day_data) - 0.5) 
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')
        
        bar_canvas.draw_idle()

    def _update_route_button_state(self):
        target_key = self._get_current_key()
        widgets = self.live_ui_components
        combo = widgets['day_selector']
        btn = widgets['view_route_btn']
        
        if not target_key or combo.count() == 0:
            btn.setEnabled(False)
            return

        try:
            day = int(combo.currentText())
            has_route = day in self.historical_routes.get(target_key, {})
            btn.setEnabled(has_route)
        except ValueError:
            btn.setEnabled(False)

    def _view_route_for_current_selection(self):
        """Wrapper to pass the current key to the logic."""
        target_key = self._get_current_key()
        if target_key:
            self._view_route_for_selected_day(target_key)

    # -------------------------------------------------------------------------
    # DATA PROCESSING
    # -------------------------------------------------------------------------
    @Slot(str)
    def _process_single_line_on_main_thread(self, line):
        self.raw_log_area.append(line.strip())
        self.parse_buffer(line)

    def parse_buffer(self, line: str) -> str:
        if 'GUI_DAY_LOG_START:' in line or 'GUI_SUMMARY_LOG_START:' in line:
            self._process_single_record(line.strip())
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

                    # Update Tracking Dictionaries
                    self.available_samples_dict[policy].add(sample)
                    
                    # [MODIFIED] Update Dropdowns dynamically
                    self._update_live_dropdowns(policy, sample)

                    data = json.loads(parts[3])
                    
                    if 'tour' in data:
                        self.historical_routes[key][day] = data['tour']
                    
                    for m, v in data.items():
                        if m in HEATMAP_METRICS: 
                            self.historical_bin_data[key][m][day] = v 
                        elif m in TARGET_METRICS or m in udef.SIM_METRICS:
                            self.daily_data[key][m][day] = float(v)
                    
                    self.status_label.setText(f"Processing: {key} day {day}")
                
                # Signal logic to process charts
                # If this incoming data matches what we are currently looking at, trigger update
                current_key = self._get_current_key()
                if not current_key:
                    # If nothing selected yet, force select this first one
                    self.live_policy_combo.setCurrentText(policy)
                    self.live_sample_combo.setCurrentText(str(sample))
                
                if current_key == key or current_key is None:
                    self.start_chart_processing.emit(key)
                    
            except Exception as e: print(f"Log Error: {e}")

        elif record.startswith("GUI_SUMMARY_LOG_START:"):
            try:
                # Find the last closing brace to ensure we get the valid JSON object
                # (SOMETIMES logs might have extra noise or multiple concatenated jsons if not careful, 
                # though usually it's one line per record)
                end = record.rfind('}')
                if end == -1: return

                clean = record[:end+1]
                json_part = clean.split("GUI_SUMMARY_LOG_START:")[1].strip()
                new_summary = json.loads(json_part)
                
                # MERGE LOGIC:
                # 1. Update/Extend Policies list (maintain order if possible, or just append unique)
                existing_policies = set(self.summary_data['policies'])
                incoming_policies = new_summary.get('policies', [])
                
                for p in incoming_policies:
                    if p not in existing_policies:
                        self.summary_data['policies'].append(p)
                        existing_policies.add(p)
                
                # 2. Update Log Data (Mean)
                # Structure: {'policy_name': [metrics...], ...}
                if 'log' in new_summary:
                    self.summary_data['log'].update(new_summary['log'])
                    
                # 3. Update Log Std Data (StdDev)
                if 'log_std' in new_summary:
                    self.summary_data['log_std'].update(new_summary['log_std'])
                    
                # 4. Update Meta info
                # We take the max n_samples or just overwrite if they are consistent/increasing
                if 'n_samples' in new_summary:
                    self.summary_data['n_samples'] = new_summary['n_samples']

                self.status_label.setText("Simulation Complete (Summary Updated).")
                self.redraw_summary_chart()
                
            except Exception as e: 
                print(f"Summary Error: {e}")
                # Print the problematic record for debugging if needed
                print(f"Problem Record: {record}")

    # -------------------------------------------------------------------------
    # UTILS & OLD LOGIC PRESERVED
    # -------------------------------------------------------------------------
    def _view_route_for_selected_day(self, target_key):
        # ... (Same logic as provided in previous file, uses target_key) ...
        widgets = self.live_ui_components
        combo = widgets['day_selector']
        selected_day_str = combo.currentText()
        if not selected_day_str: return
        
        day = int(selected_day_str)
        route_points = self.historical_routes.get(target_key, {}).get(day, [])
        
        if not route_points: return

        # ... (Folium logic remains identical to your provided file) ...
        # (Shortened here for brevity, assume full Folium implementation)
        points_to_plot = [p for p in route_points if 'lat' in p and 'lng' in p]
        if not points_to_plot: return

        depot_loc = next(( (p['lat'],p['lng']) for p in points_to_plot if p.get('type')=='depot'), 
                         (points_to_plot[0]['lat'], points_to_plot[0]['lng']))
        
        m = folium.Map(location=depot_loc, zoom_start=13)
        poly_coords = []
        for p in points_to_plot:
            lat, lng = p['lat'], p['lng']
            poly_coords.append((lat, lng))
            icon = folium.Icon(color='green', icon='home') if p.get('type')=='depot' else None
            if not icon:
                folium.CircleMarker((lat,lng), radius=6, color='gray', fill=True, popup="Bin").add_to(m)
            else:
                folium.Marker((lat,lng), popup="Depot", icon=icon).add_to(m)
        
        if len(poly_coords) > 1:
            folium.PolyLine(poly_coords, color='blue', weight=2).add_to(m)
            
        try:
            temp_root = os.path.join(udef.ROOT_DIR, 'temp')
            os.makedirs(temp_root, exist_ok=True)
            temp_filename = f"route_{target_key.replace(' ', '_')}_{day}.html"
            temp_path = os.path.join(temp_root, temp_filename)
            m.save(temp_path)
            webbrowser.open('file://' + os.path.abspath(temp_path))
        except Exception as e:
            print(f"Map Error: {e}")

    def _on_main_tab_changed(self, index):
        if self.tabs.tabText(index) == "Average and StdDev (Summary)":
            self._populate_summary_policy_combo()

    def _populate_summary_policy_combo(self):
        self.summary_policy_combo.blockSignals(True)
        self.summary_policy_combo.clear()
        policies = sorted(self.available_samples_dict.keys())
        self.summary_policy_combo.addItems(policies)
        self.summary_policy_combo.blockSignals(False)
        if policies: self._populate_summary_sample_combo()

    def _populate_summary_sample_combo(self):
        policy = self.summary_policy_combo.currentText()
        if not policy: return
        self.summary_sample_combo.blockSignals(True)
        self.summary_sample_combo.clear()
        samples = sorted(list(self.available_samples_dict.get(policy, [])))
        self.summary_sample_combo.addItems([str(s) for s in samples])
        self.summary_sample_combo.blockSignals(False)
        if samples: self._draw_selected_summary_heatmap()

    def _draw_selected_summary_heatmap(self):
        # ... (Same logic as provided file) ...
        policy = self.summary_policy_combo.currentText()
        sample_str = self.summary_sample_combo.currentText()
        if not policy or not sample_str: return
        target_key = f"{policy} sample {sample_str}"
        
        self.hm_summary_fig.clear()
        
        with QMutexLocker(self.data_mutex):
            hist_data = self.historical_bin_data.get(target_key, {})
            if not hist_data: return

            axes = self.hm_summary_fig.subplots(len(HEATMAP_METRICS), 1, sharex=True)
            if len(HEATMAP_METRICS) == 1: axes = [axes]
            self.hm_summary_fig.tight_layout(pad=3.0)

            for i, metric in enumerate(HEATMAP_METRICS):
                ax = axes[i]
                metric_history = hist_data.get(metric, {})
                days = sorted(metric_history.keys())
                if not days: continue
                mat = np.array([metric_history[d] for d in days])
                cmap = 'hot' if metric in ['bin_state_c', 'bins_state_c_after'] else 'viridis'
                ax.imshow(mat, aspect='auto', cmap=cmap, origin='lower')
                ax.set_title(metric)
                ax.set_ylabel("Day")
            axes[-1].set_xlabel("Bin Index")
        self.hm_summary_canvas.draw_idle()

    def redraw_summary_chart(self):
        # ... (Same logic as provided file) ...
        if not self.summary_data: return
        self.summary_ax.clear()
        log = self.summary_data['log']
        log_std = self.summary_data['log_std']
        policy_names = self.summary_data['policies']
        colors = self._generate_distinct_colors(len(policy_names))
        selection = self.summary_metric_combo.currentText()

        if selection == "All Metrics":
            n_metrics = len(udef.SIM_METRICS)
            bar_width = 0.8 / len(policy_names)
            x = np.arange(n_metrics)
            for i, policy in enumerate(policy_names):
                means = [log[policy][j] for j in range(n_metrics)]
                stds = [log_std[policy][j] for j in range(n_metrics)]
                self.summary_ax.bar(x + bar_width * i, means, width=bar_width, yerr=stds, label=policy, color=colors[i])
            self.summary_ax.legend()
            self.summary_ax.set_xticks(x + bar_width * (len(policy_names)-1)/2, udef.SIM_METRICS)
        else:
            if selection not in udef.SIM_METRICS: return
            idx = udef.SIM_METRICS.index(selection)
            
            # --- START OF NECESSARY CHANGES ---
            num_policies = len(policy_names)
            x_pos = np.arange(num_policies) # Array of positions [0, 1, 2, ...]

            means = [log[p][idx] for p in policy_names]
            stds = [log_std[p][idx] for p in policy_names]

            # Plot the bars at the positions x_pos
            self.summary_ax.bar(x_pos, means, yerr=stds, color=colors)

            # Explicitly set the tick locations to match the bar positions
            self.summary_ax.set_xticks(x_pos)

            # Set the labels at those explicit tick locations
            # The rotation is added to prevent labels from overlapping
            self.summary_ax.set_xticklabels(policy_names, rotation=25, ha='right')
            # --- END OF NECESSARY CHANGES ---
            
            # print(policy_names) # You can remove this print statement
            
        self.summary_canvas.draw_idle()

    def setup_summary_area(self):
        # ... (Same logic as provided file) ...
        self.summary_tab = QWidget(); self.summary_layout = QVBoxLayout(self.summary_tab)
        self.summary_nested_tabs = QTabWidget()
        self.summary_layout.addWidget(self.summary_nested_tabs)

        # Tab 1
        self.scalar_summary_widget = QWidget(); scalar_layout = QVBoxLayout(self.scalar_summary_widget)
        ctl = QHBoxLayout()
        self.summary_metric_combo = QComboBox(); self.summary_metric_combo.addItem("All Metrics"); self.summary_metric_combo.addItems(udef.SIM_METRICS)
        self.summary_metric_combo.currentTextChanged.connect(self.redraw_summary_chart)
        ctl.addWidget(QLabel("Metric:")); ctl.addWidget(self.summary_metric_combo); ctl.addStretch()
        scalar_layout.addLayout(ctl)
        self.summary_fig = Figure(figsize=(10,6)); self.summary_canvas = FigureCanvas(self.summary_fig); self.summary_ax = self.summary_fig.add_subplot(111)
        scalar_layout.addWidget(self.summary_canvas)
        self.summary_nested_tabs.addTab(self.scalar_summary_widget, "Metrics")

        # Tab 2
        self.hm_summary_widget = QWidget(); hm_layout = QVBoxLayout(self.hm_summary_widget)
        ctl2 = QHBoxLayout()
        self.summary_policy_combo = QComboBox(); self.summary_policy_combo.currentIndexChanged.connect(self._populate_summary_sample_combo)
        self.summary_sample_combo = QComboBox(); self.summary_sample_combo.currentIndexChanged.connect(self._draw_selected_summary_heatmap)
        ctl2.addWidget(QLabel("Policy:")); ctl2.addWidget(self.summary_policy_combo)
        ctl2.addWidget(QLabel("Sample:")); ctl2.addWidget(self.summary_sample_combo); ctl2.addStretch()
        hm_layout.addLayout(ctl2)
        self.hm_summary_fig = Figure(figsize=(10,8)); self.hm_summary_canvas = FigureCanvas(self.hm_summary_fig)
        hm_layout.addWidget(self.hm_summary_canvas)
        self.summary_nested_tabs.addTab(self.hm_summary_widget, "Heatmaps")
        self.tabs.addTab(self.summary_tab, "Average and StdDev (Summary)")

    def setup_raw_log_area(self):
        self.raw_tab = QWidget(); self.raw_layout = QVBoxLayout(self.raw_tab)
        self.tabs.addTab(self.raw_tab, "Raw Output (Log)")
        self.raw_log_area = QTextEdit()
        self.raw_log_area.setReadOnly(True)
        self.raw_layout.addWidget(self.raw_log_area)

    def _generate_distinct_colors(self, num_colors):
        return [f"#{random.randint(0, 0xFFFFFF):06x}" for _ in range(num_colors)]

    def stop_thread(self):
        print("Stopping threads...")
        self.file_tailer.stop()
        self.file_thread.quit(); self.file_thread.wait()
        self.chart_thread.quit(); self.chart_thread.wait()

    def closeEvent(self, event):
        self.stop_thread()
        event.accept()