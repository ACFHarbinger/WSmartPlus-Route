import json
import random
import numpy as np

from collections import defaultdict
from PySide6.QtWidgets import (
    QTabWidget, QLabel, QTextEdit,
    QWidget, QVBoxLayout, QSizePolicy,
)
from PySide6.QtCore import (
    Qt, Signal, QThread, 
    Slot, QMutex, QMutexLocker 
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator

from ..helpers import ChartWorker 
from ..app_definitions import TARGET_METRICS, SUMMARY_METRICS, HEATMAP_METRICS


class SimulationResultsWindow(QWidget):
    """
    Window with tabs for Raw Log, Summary, and individual (Policy, Sample) charts.
    """
    start_chart_processing = Signal(str) 

    def __init__(self, policy_names):
        super().__init__()
        self.setWindowTitle("Simulation Chart and Raw Output (Per Sample)")
        self.setWindowFlags(self.windowFlags() | Qt.Window) 

        self.data_mutex = QMutex() 
        
        self.daily_data = defaultdict(lambda: defaultdict(dict))
        
        # New: Store the LATEST snapshot of bin data for heatmaps
        self.latest_bin_data = defaultdict(dict)
        self.historical_bin_data = defaultdict(lambda: defaultdict(dict))
        
        self.policy_names = policy_names
        self.active_sample_keys = set() 
        self.summary_data = {} 
        self.is_complete = False
        self.policy_chart_widgets = {} 
        
        main_layout = QVBoxLayout(self)
        self.status_label = QLabel("Waiting for simulation to start...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(self.status_label)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.policy_tabs_container = QTabWidget() 
        self.tabs.addTab(self.policy_tabs_container, "Metric Evolution (Charts)")
        
        self.setup_raw_log_area()
        self.setup_summary_chart()

        self.plot_color = "#3465a4" 
        
        # --- THREAD SETUP ---
        self.chart_thread = QThread()
        self.chart_worker = ChartWorker(
            daily_data=self.daily_data, 
            metrics_to_plot=TARGET_METRICS,
            data_mutex=self.data_mutex 
        )
        self.chart_worker.moveToThread(self.chart_thread)
        self.chart_thread.start()
        
        self.start_chart_processing.connect(self.chart_worker.process_data)
        self.chart_worker.data_ready.connect(self._update_chart_on_main_thread) 
        
    def stop_thread(self):
        print("Stopping chart worker thread...")
        self.chart_thread.quit()
        self.chart_thread.wait()
        print("Thread stopped.")

    def closeEvent(self, event):
        self.stop_thread()
        event.accept() 

    @Slot(str, dict)
    def _update_chart_on_main_thread(self, target_key, processed_data):
        """
        Updates Line Charts (Time Series) AND Heatmaps (Current State).
        """
        if target_key not in self.policy_chart_widgets:
            return

        # 1. Update Line Charts
        chart_widgets = self.policy_chart_widgets[target_key]
        line_axes = chart_widgets['line_axes']
        line_canvas = chart_widgets['line_canvas']
        max_days = processed_data['max_days']

        for i, metric in enumerate(TARGET_METRICS):
            ax = line_axes[i]
            ax.clear() 
            ax.grid(True)
            ax.set_title(f'{metric} for {target_key}')
            ax.set_ylabel(metric)

            all_values_for_metric = []
            metric_data = processed_data['metrics'].get(metric)
            
            if metric_data and metric_data['days']:
                days = metric_data['days']
                values = metric_data['values']
                all_values_for_metric.extend(values)
                ax.plot(days, values, marker='o', markersize=3, linestyle='-', color=self.plot_color)

            if all_values_for_metric:
                min_val = min(all_values_for_metric)
                max_val = max(all_values_for_metric)
                y_range = max_val - min_val
                
                if y_range == 0:
                    if max_val == 0: ax.set_ylim(-0.1, 1.1)
                    else:
                        buffer = max(1.0, abs(max_val * 0.1))
                        ax.set_ylim(min_val - buffer, max_val + buffer)
                else:
                    buffer = y_range * 0.05
                    ax.set_ylim(min_val - buffer, max_val + buffer)
            else:
                ax.set_ylim(-0.1, 1.1) 

            if max_days > 0:
                ax.set_xlim(0.8, max_days + 0.2)
            else:
                ax.set_xlim(0, 100)
            
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            ax.xaxis.set_minor_locator(plt.NullLocator()) 

            if i == len(TARGET_METRICS) - 1:
                ax.set_xlabel("Day")

        line_canvas.draw_idle()

        # 2. Update Heatmaps (Bin State)
        self._update_heatmaps(target_key)

    def _update_heatmaps(self, target_key):
        """Draws the 2D heatmaps (Day vs Bin Index) based on historical_bin_data."""
        if target_key not in self.historical_bin_data or not self.historical_bin_data[target_key]:
            return
            
        widgets = self.policy_chart_widgets[target_key]
        hm_axes = widgets['hm_axes']
        hm_canvas = widgets['hm_canvas']
        info_label = widgets['hm_info_label']
        
        # Fetch latest global scalar data
        hist_data = self.historical_bin_data[target_key]
        travel = hist_data.get('bin_state_travel', 0)
        ndays = hist_data.get('bin_state_ndays', 0)
        info_label.setText(f"<b>Simulation Day:</b> {ndays} | <b>Total Travel:</b> {travel:.2f} km")

        # Define titles for the heatmap rows
        titles = {
            'bin_state_c': "Current Fill Level (%)",
            'bin_state_collected': "Total Waste Collected (kg)",
        }
        
        # --- Draw Heatmaps (Day vs Bin Index) ---
        for i, metric in enumerate(HEATMAP_METRICS):
            ax = hm_axes[i]
            ax.clear()
            
            metric_history = hist_data.get(metric)
            
            if metric_history:
                days = sorted(metric_history.keys())
                stacked_data = np.array([metric_history[d] for d in days]) 
                
                # Determine colormap and normalization
                cmap = 'viridis'
                vmin, vmax = None, None
                
                if metric == 'bin_state_c':
                    cmap = 'RdYlGn_r' # Green low, Red high
                    # Set vmin and vmax closer to the expected average (50%)
                    vmin, vmax = 30, 70  # Adjust this range (e.g., 30-70 or 40-60)
                
                # Use imshow, setting extent and origin for correct axis labels.
                im = ax.imshow(
                    stacked_data, 
                    aspect='auto', 
                    cmap=cmap, 
                    vmin=vmin, 
                    vmax=vmax,
                    origin='lower'
                )
                
                # Set X-Axis: Bin Index
                ax.set_xlabel("Bin Index")
                ax.set_xticks(np.arange(stacked_data.shape[1]))
                
                # Set Y-Axis: Day
                ax.set_ylabel("Day")
                ax.set_yticks(np.arange(len(days)))
                ax.set_yticklabels(days) # Label ticks with actual day numbers
                
                ax.set_title(titles[metric])
                
                # Add colorbar (only if not already present, or managed properly)
                # If the figure is being cleared and redrawn, adding a colorbar dynamically is best done outside the axis object.
                # For simplicity, we skip full colorbar management here, relying on the visual representation of the map.
            else:
                ax.text(0.5, 0.5, "No Data", ha='center', va='center')

        # Ensure the figure redraws after all subplots are updated
        hm_canvas.draw_idle()

    def setup_raw_log_area(self):
        self.raw_tab = QWidget(); self.raw_layout = QVBoxLayout(self.raw_tab)
        self.tabs.addTab(self.raw_tab, "Raw Output (Log)")
        self.raw_log_area = QTextEdit()
        self.raw_log_area.setReadOnly(True)
        self.raw_log_area.setStyleSheet("font-family: monospace; font-size: 10pt; background: #2e3436; color: #d3d7cf;")
        self.raw_layout.addWidget(self.raw_log_area)
        
    def setup_summary_chart(self):
        self.summary_tab = QWidget(); self.summary_layout = QVBoxLayout(self.summary_tab)
        self.tabs.addTab(self.summary_tab, "Average and StdDev (Summary)")
        self.summary_fig = Figure(figsize=(10, 6))
        self.summary_canvas = FigureCanvas(self.summary_fig)
        self.summary_layout.addWidget(self.summary_canvas)
        self.summary_ax = self.summary_fig.add_subplot(111)
        self.summary_fig.tight_layout(pad=1.0)
        self.summary_ax.set_title("Simulation Summary (Averages)")
        self.summary_fig.canvas.draw()
        
    def add_sample_chart_tab(self, policy_sample_key):
        """
        Creates a tab containing a SUB-TAB widget:
        Tab 1: Time Series (Lines)
        Tab 2: Bin State (Heatmaps)
        """
        
        main_container_widget = QWidget()
        main_layout = QVBoxLayout(main_container_widget)
        
        # Sub-tabs within the Policy Tab
        sub_tabs = QTabWidget()
        main_layout.addWidget(sub_tabs)
        
        # --- TAB 1: LINE CHARTS ---
        line_tab = QWidget()
        line_layout = QVBoxLayout(line_tab)
        line_fig = Figure(figsize=(10, 6))
        line_canvas = FigureCanvas(line_fig)
        line_axes = line_fig.subplots(len(TARGET_METRICS), 1, sharex=True)
        line_fig.tight_layout(pad=2.0)
        line_layout.addWidget(line_canvas)
        
        # Initialize styling for line charts
        for i, metric in enumerate(TARGET_METRICS):
            ax = line_axes[i] if len(TARGET_METRICS) > 1 else line_axes
            ax.set_title(f'{metric} for {policy_sample_key}')
            ax.grid(True)
            ax.set_xlim(0, 100) 
            ax.set_ylim(-0.1, 1.1)
            if i == len(TARGET_METRICS) - 1:
                ax.set_xlabel("Day")
        
        sub_tabs.addTab(line_tab, "Evolution (Time Series)")

        # --- TAB 2: HEATMAPS ---
        hm_tab = QWidget()
        hm_layout = QVBoxLayout(hm_tab)
        
        # Info Label
        hm_info_label = QLabel("Simulation Day: 0 | Total Travel: 0")
        hm_info_label.setStyleSheet("font-size: 14px; padding: 5px;")
        hm_layout.addWidget(hm_info_label)

        hm_fig = Figure(figsize=(10, 6))
        hm_canvas = FigureCanvas(hm_fig)
        hm_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        
        # Create 2 Rows for Fill and Collected (len(HEATMAP_METRICS))
        hm_axes = hm_fig.subplots(len(HEATMAP_METRICS), 1, sharex=True)
        
        hm_fig.tight_layout(pad=3.0)
        hm_layout.addWidget(hm_canvas)
        
        sub_tabs.addTab(hm_tab, "Bin State (Heatmaps)")

        # Store references
        self.policy_chart_widgets[policy_sample_key] = {
            'line_fig': line_fig,
            'line_canvas': line_canvas,
            'line_axes': line_axes if len(TARGET_METRICS) > 1 else [line_axes],
            
            'hm_fig': hm_fig,
            'hm_canvas': hm_canvas,
            'hm_axes': hm_axes,
            'hm_info_label': hm_info_label
        }

        self.policy_tabs_container.addTab(main_container_widget, policy_sample_key)
        self.active_sample_keys.add(policy_sample_key)

    # --- (Colors/Parsing logic) ---
    def _generate_distinct_colors(self, num_colors):
        hex_colors = []
        for i in range(num_colors):
            h = i * (360 / num_colors)
            s = 90 + random.randint(-10, 10)
            l = 50 + random.randint(-10, 10)
            r, g, b = self._hsl_to_rgb(h, s, l)
            hex_colors.append(f"#{int(r):02x}{int(g):02x}{int(b):02x}")
        return hex_colors

    def _hsl_to_rgb(self, h, s, l):
        h /= 360; s /= 100; l /= 100
        if s == 0: return l * 255, l * 255, l * 255
        def hue_to_rgb(p, q, t):
            if t < 0: t += 1;
            if t > 1: t -= 1;
            if t < 1/6: return p + (q - p) * 6 * t
            if t < 1/2: return q
            if t < 2/3: return p + (q - p) * (2/3 - t) * 6
            return p
        q = l * (1 + s) if l < 0.5 else l + s - l * s
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        return r * 255, g * 255, b * 255

    def parse_buffer(self, buffer: str) -> str:
        self.raw_log_area.append(buffer)
        if self.raw_log_area.document().blockCount() > 500:
             self.raw_log_area.clear()
             self.raw_log_area.append("--- Log Cleared (Overflow Prevention) ---")
        
        records = buffer.split('GUI_')
        processed_length = 0
        for i in range(1, len(records)):
            full_record_segment = 'GUI_' + records[i]
            
            if '}' in full_record_segment: 
                self._process_single_record(full_record_segment)
                processed_length += len(full_record_segment)
            else:
                return buffer[processed_length:]
        return "" 

    def _process_single_record(self, record):
        if record.startswith("GUI_DAY_LOG_START:"):
            end_index = record.rfind('}')
            if end_index == -1: return

            clean_record = record[:end_index + 1]
            
            try:
                policy_sample_key = "" 
                
                with QMutexLocker(self.data_mutex): 
                    parts = clean_record.split("GUI_DAY_LOG_START:")[1].strip().split(',', 3)
                    if len(parts) != 4: raise ValueError(f"Malformed record structure.")
                    
                    policy = parts[0].strip().replace('\r', '').replace('\n', '') 
                    sample_idx = int(parts[1].strip())
                    day = int(parts[2].strip())
                    
                    policy_sample_key = f"{policy} sample {sample_idx}"

                    if policy_sample_key not in self.active_sample_keys:
                        self.add_sample_chart_tab(policy_sample_key)
                    
                    json_part = parts[3].strip()
                    start_json = json_part.find('{')
                    end_json = json_part.rfind('}')
                    
                    if start_json == -1 or end_json == -1 or end_json < start_json: raise ValueError(f"JSON payload not found.")
                    
                    json_string_only = json_part[start_json : end_json + 1]
                    metrics = json.loads(json_string_only)
                    
                    # Store data using the unique key
                    for metric, value in metrics.items():
                        # Handle scalar data (Line Charts)
                        if metric in TARGET_METRICS or metric in SUMMARY_METRICS:
                            self.daily_data[policy_sample_key][metric][day] = float(value) 
                        
                        # [NEW/MODIFIED] Handle array data (Heatmaps: Day vs Bin)
                        elif metric in HEATMAP_METRICS:
                            # Store the list of bin values under the current day
                            self.historical_bin_data[policy_sample_key][metric][day] = value
                        
                        # [NEW] Handle scalar global state data (for text updates)
                        elif metric in ['bin_state_travel', 'bin_state_ndays']:
                             # Store the latest global state data
                             self.historical_bin_data[policy_sample_key][metric] = value
                    
                    self.status_label.setText(f"Processing: {policy_sample_key} day {day}")
                
                if policy_sample_key:
                    self.start_chart_processing.emit(policy_sample_key) 
                
            except Exception as e:
                print(f"CRITICAL PARSING ERROR (Day Log): {e} in record: {clean_record}")
        
        elif record.startswith("GUI_SUMMARY_LOG_START:"):
            end_index = record.rfind('}')
            if end_index == -1: return
                
            clean_record = record[:end_index + 1]
            try:
                json_part = clean_record.split("GUI_SUMMARY_LOG_START:")[1].strip()
                start_json = json_part.find('{')
                end_json = json_part.rfind('}')
                
                if start_json == -1 or end_json == -1 or end_json < start_json: raise ValueError(f"JSON payload not found.")
                
                json_string_only = json_part[start_json : end_json + 1]
                summary_data = json.loads(json_string_only)
                
                self.summary_data = summary_data
                self.is_complete = True
                self.status_label.setText("Simulation Complete. Final results loaded.")
                self.redraw_summary_chart()
            except Exception as e:
                print(f"CRITICAL PARSING ERROR (Summary Log): {e} in record: {clean_record}")

    def redraw_summary_chart(self):
        if not self.summary_data:
            self.summary_ax.text(0.5, 0.5, "No summary data available.", 
                                 transform=self.summary_ax.transAxes, ha='center')
            return

        self.summary_ax.clear()
        log = self.summary_data['log']
        log_std = self.summary_data['log_std']
        metric_labels = SUMMARY_METRICS
        policy_names = self.summary_data['policies']
        n_policies = len(policy_names)
        n_metrics = len(metric_labels)
        bar_width = 0.8 / n_policies
        x = np.arange(n_metrics)
        summary_colors = self._generate_distinct_colors(n_policies)

        for i, policy in enumerate(policy_names):
            means = [log[policy][j] for j in range(n_metrics)]
            stds = [log_std[policy][j] for j in range(n_metrics)]
            r = x + bar_width * i
            self.summary_ax.bar(r, means, width=bar_width, 
                                edgecolor='grey', label=policy,
                                yerr=stds, capsize=5, color=summary_colors[i % len(summary_colors)])

        self.summary_ax.set_xlabel("Metrics")
        self.summary_ax.set_ylabel("Mean Value")
        self.summary_ax.set_title("Average and Standard Deviation per Metric (Across Samples)")
        self.summary_ax.set_xticks(x + bar_width * (n_policies - 1) / 2, metric_labels)
        self.summary_ax.tick_params(axis='x', rotation=45)
        self.summary_ax.legend()
        self.summary_fig.canvas.draw_idle()