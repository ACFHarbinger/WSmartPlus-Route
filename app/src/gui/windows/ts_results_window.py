import json
import random
import numpy as np

from collections import defaultdict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, 
    QTabWidget, QLabel, QTextEdit
)
from PySide6.QtCore import (
    Qt, Signal, QThread, Slot, QMutex, QMutexLocker 
)
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.ticker import MaxNLocator
from ..helpers import ChartWorker 
from ..app_definitions import TARGET_METRICS, SUMMARY_METRICS


class SimulationResultsWindow(QWidget):
    """
    Window with tabs for Raw Log, Summary, and individual (Policy, Sample) charts.
    """
    # Signal to start the worker's data processing
    start_chart_processing = Signal(str) 

    def __init__(self, policy_names):
        super().__init__()
        self.setWindowTitle("Simulation Chart and Raw Output (Per Sample)")
        self.setWindowFlags(self.windowFlags() | Qt.Window) 

        self.data_mutex = QMutex() 
        
        self.daily_data = defaultdict(lambda: defaultdict(dict)) 
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
        
        # 1. Main thread signals worker to start processing
        self.start_chart_processing.connect(self.chart_worker.process_data)
        
        # 2. Worker signals main thread with processed data
        self.chart_worker.data_ready.connect(self._update_chart_on_main_thread) 
        
    def stop_thread(self):
        """Safely stops the worker thread."""
        print("Stopping chart worker thread...")
        self.chart_thread.quit()
        self.chart_thread.wait() # Wait for thread to finish
        print("Thread stopped.")

    def closeEvent(self, event):
        """Overrides QWidget.closeEvent to safely stop the worker thread."""
        self.stop_thread()
        event.accept() 

    @Slot(str, dict)
    def _update_chart_on_main_thread(self, target_key, processed_data):
        """
        Slot executed on the MAIN thread to safely perform all Matplotlib operations.
        """
        if target_key not in self.policy_chart_widgets:
            return

        chart_data = self.policy_chart_widgets[target_key]
        axes = chart_data['axes']
        canvas = chart_data['canvas']
        max_days = processed_data['max_days']

        # --- All Matplotlib logic is now safely on the Main Thread ---
        for i, metric in enumerate(TARGET_METRICS):
            ax = axes[i]
            ax.clear() 
            ax.grid(True)
            ax.set_title(f'{metric} for {target_key}')
            ax.set_ylabel(metric)

            all_values_for_metric = []
            
            # Get processed data from the worker
            metric_data = processed_data['metrics'].get(metric)
            
            if metric_data and metric_data['days']:
                days = metric_data['days']
                values = metric_data['values']
                all_values_for_metric.extend(values)
                
                # Plot the single line
                ax.plot(days, values, 
                        marker='o', markersize=3, linestyle='-', color=self.plot_color)

            # 3. Set Dynamic Y-Axis Limits
            if all_values_for_metric:
                min_val = min(all_values_for_metric)
                max_val = max(all_values_for_metric)
                y_range = max_val - min_val
                
                if y_range == 0:
                    if max_val == 0: ax.set_ylim(-0.1, 1.1)
                    else:
                        buffer = max(1.0, abs(max_val * 0.1)) # Ensure buffer is at least 1
                        ax.set_ylim(min_val - buffer, max_val + buffer)
                else:
                    buffer = y_range * 0.05
                    ax.set_ylim(min_val - buffer, max_val + buffer)
            else:
                ax.set_ylim(-0.1, 1.1) 

            # 4. Set X-Axis Limits and Ticks
            if max_days > 0:
                ax.set_xlim(0.8, max_days + 0.2)
            else:
                ax.set_xlim(0, 100)
            
            # Use Matplotlib's automatic integer locator
            ax.xaxis.set_major_locator(MaxNLocator(integer=True, min_n_ticks=1))
            
            # Explicitly disable minor ticks to prevent minorTicks[0] crash
            ax.xaxis.set_minor_locator(plt.NullLocator()) 

            if i == len(TARGET_METRICS) - 1:
                ax.set_xlabel("Day")

        # 5. Final draw call
        canvas.draw_idle()

    # --- Setup methods (Unchanged) ---
    def setup_raw_log_area(self):
        """Adds the Raw Output tab"""
        self.raw_tab = QWidget(); self.raw_layout = QVBoxLayout(self.raw_tab)
        self.tabs.addTab(self.raw_tab, "Raw Output (Log)")
        self.raw_log_area = QTextEdit()
        self.raw_log_area.setReadOnly(True)
        self.raw_log_area.setStyleSheet("font-family: monospace; font-size: 10pt; background: #2e3436; color: #d3d7cf;")
        self.raw_layout.addWidget(self.raw_log_area)
        
    def setup_summary_chart(self):
        """Adds the Summary tab"""
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
        """Dynamically creates a tab for a new (policy, sample) pair."""
        
        policy_widget = QWidget()
        policy_layout = QVBoxLayout(policy_widget)
        
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        
        axes = fig.subplots(len(TARGET_METRICS), 1, sharex=True)
        fig.tight_layout(pad=2.0)
        
        self.policy_chart_widgets[policy_sample_key] = {
            'fig': fig,
            'canvas': canvas,
            'axes': axes if len(TARGET_METRICS) > 1 else [axes], 
        }
        policy_layout.addWidget(canvas)
        
        for i, metric in enumerate(TARGET_METRICS):
            ax = self.policy_chart_widgets[policy_sample_key]['axes'][i]
            ax.set_title(f'{metric} for {policy_sample_key}')
            ax.grid(True)
            ax.set_xlim(0, 100) 
            ax.set_ylim(-0.1, 1.1)
            if i == len(TARGET_METRICS) - 1:
                ax.set_xlabel("Day")

        self.policy_tabs_container.addTab(policy_widget, policy_sample_key)
        fig.canvas.draw() 
        self.active_sample_keys.add(policy_sample_key)


    # --- Utility methods (Omitted) ---
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


    # --- Parsing methods (Modified) ---
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
                policy_sample_key = "" # Define scope
                
                # --- CRITICAL SECTION: DATA WRITING (Protected) ---
                with QMutexLocker(self.data_mutex): 
                    
                    parts = clean_record.split("GUI_DAY_LOG_START:")[1].strip().split(',', 3)
                    if len(parts) != 4: raise ValueError(f"Malformed record structure.")
                    
                    policy = parts[0].strip().replace('\r', '').replace('\n', '') 
                    sample_idx = int(parts[1].strip())
                    day = int(parts[2].strip())
                    
                    # Create the unique key for this tab/chart
                    policy_sample_key = f"{policy} sample {sample_idx}"

                    # Check if we need to create a new tab for this sample
                    if policy_sample_key not in self.active_sample_keys:
                        # This is safe because we are on the main thread
                        self.add_sample_chart_tab(policy_sample_key)
                    
                    json_part = parts[3].strip()
                    start_json = json_part.find('{')
                    end_json = json_part.rfind('}')
                    
                    if start_json == -1 or end_json == -1 or end_json < start_json: raise ValueError(f"JSON payload not found.")
                    
                    json_string_only = json_part[start_json : end_json + 1]
                    metrics = json.loads(json_string_only)
                    
                    for metric, value in metrics.items():
                        float_value = float(value)
                        
                        # Store data using the unique key
                        if metric in TARGET_METRICS or metric in SUMMARY_METRICS:
                            self.daily_data[policy_sample_key][metric][day] = float_value 
                    
                    self.status_label.setText(f"Processing: {policy_sample_key} day {day}")
                # --- END CRITICAL SECTION ---
                
                # Emit the unique key to the worker for processing
                if policy_sample_key:
                    self.start_chart_processing.emit(policy_sample_key) 
                
            except Exception as e:
                print(f"CRITICAL PARSING ERROR (Day Log): {e} in record: {clean_record}")
        
        elif record.startswith("GUI_SUMMARY_LOG_START:"):
            # This logic remains unchanged
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
        """Redraws the summary bar chart."""
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