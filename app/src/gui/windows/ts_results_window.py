# --- File: ts_results_window.py (REBUILT WITH QTIMER) ---
import json
import random
import numpy as np

from collections import defaultdict
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, 
    QTabWidget, QLabel,
)
from PySide6.QtCore import Qt, Signal, QThread, Slot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from ..helpers import ChartWorker
try:
    from app.src.utils.definitions import METRICS
except ImportError:
    METRICS = ['overflows', 'kg', 'ncol', 'km', 'kg/km', 'cost'] 



class SimulationResultsWindow(QWidget):
    # Signal emitted from main thread (in parse_buffer) to trigger the worker's slot
    start_chart_update = Signal(str)

    def __init__(self, policy_names):
        super().__init__()
        self.setWindowTitle("Simulation Results Dashboard")
        self.setMinimumSize(1200, 800)
        self.setWindowFlags(self.windowFlags() | Qt.Window) 

        self.daily_data = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
        self.sample_counts = defaultdict(int) 
        self.summary_data = {} 
        self.policy_names = policy_names
        self.is_complete = False
        self.policy_chart_widgets = {} 
        
        main_layout = QVBoxLayout(self)
        self.status_label = QLabel("Waiting for simulation to start...")
        self.status_label.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(self.status_label)
        
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        self.per_sample_tab = QWidget()
        self.per_sample_layout = QVBoxLayout(self.per_sample_tab)
        self.tabs.addTab(self.per_sample_tab, "Daily Metric Evolution")
        self.setup_daily_charts()

        self.summary_tab = QWidget()
        self.summary_layout = QVBoxLayout(self.summary_tab)
        self.tabs.addTab(self.summary_tab, "Average & StdDev Summary")
        self.setup_summary_chart()

        self.colors = self._generate_distinct_colors(20) 
        
        # --- THREAD SETUP ---
        self.chart_thread = QThread()
        self.chart_worker = ChartWorker(
            daily_data=self.daily_data, 
            sample_counts=self.sample_counts, 
            policy_chart_widgets=self.policy_chart_widgets, 
            colors=self.colors
        )
        self.chart_worker.moveToThread(self.chart_thread)
        self.chart_thread.start()
        
        # Connection 1: Main Thread Data Ready -> Worker Thread Update Figure (Cross-thread)
        self.start_chart_update.connect(self.chart_worker.update_figure)
        
        # Connection 2: Worker Thread Draw Request -> Main Thread Draw Canvas (Cross-thread GUI update)
        self.chart_worker.draw_request.connect(self._redraw_canvas_on_main_thread)
        
        self.destroyed.connect(self.stop_thread)
        
    def stop_thread(self):
        """Safely stops the worker thread."""
        self.chart_thread.quit()
        self.chart_thread.wait()

    @Slot(object)
    def _redraw_canvas_on_main_thread(self, canvas):
        """
        Slot executed on the MAIN thread to safely trigger Matplotlib rendering.
        This is the only place canvas.draw_idle() should be called.
        """
        canvas.draw_idle() 

    # --- Utility methods (unchanged) ---
    def _generate_distinct_colors(self, num_colors):
        """Generates a list of distinct colors."""
        hex_colors = []
        for i in range(num_colors):
            h = i * (360 / num_colors)
            s = 90 + random.randint(-10, 10)
            l = 50 + random.randint(-10, 10)
            r, g, b = self._hsl_to_rgb(h, s, l)
            hex_colors.append(f"#{int(r):02x}{int(g):02x}{int(b):02x}")
        return hex_colors

    def _hsl_to_rgb(self, h, s, l):
        """Converts HSL to RGB."""
        h /= 360
        s /= 100
        l /= 100

        if s == 0:
            return l * 255, l * 255, l * 255

        def hue_to_rgb(p, q, t):
            if t < 0: t += 1
            if t > 1: t -= 1
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

    # --- Setup methods ---
    def setup_daily_charts(self):
        """Initializes a QTabWidget inside the Daily tab, one sub-tab per policy."""
        self.policy_tabs = QTabWidget()
        self.per_sample_layout.addWidget(self.policy_tabs)
        
        for policy in self.policy_names:
            policy_widget = QWidget()
            policy_layout = QVBoxLayout(policy_widget)
            
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            
            axes = fig.subplots(len(METRICS), 1, sharex=True)
            fig.tight_layout(pad=1.0)
            
            self.policy_chart_widgets[policy] = {
                'fig': fig,
                'canvas': canvas,
                'axes': axes,
                'next_sample_color_idx': 0
            }
            policy_layout.addWidget(canvas)
            
            for i, metric in enumerate(METRICS):
                ax = axes[i]
                ax.set_title(f'{metric} for {policy}')
                ax.grid(True)
                ax.set_xlim(0, 100) 
                if i == len(METRICS) - 1:
                    ax.set_xlabel("Day")

            self.policy_tabs.addTab(policy_widget, policy)
            # Initial draw is safe here
            fig.canvas.draw() 

    def setup_summary_chart(self):
        """Initializes the Matplotlib canvas for summary bar plotting."""
        self.summary_fig = Figure(figsize=(10, 6))
        self.summary_canvas = FigureCanvas(self.summary_fig)
        self.summary_layout.addWidget(self.summary_canvas)
        self.summary_ax = self.summary_fig.add_subplot(111)
        self.summary_fig.tight_layout(pad=1.0)
        self.summary_ax.set_title("Simulation Summary (Averages)")
        self.summary_fig.canvas.draw()


    # --- Parsing methods (Triggers QThread) ---
    def parse_buffer(self, buffer: str) -> str:
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
            if end_index == -1:
                 print(f"CRITICAL PARSING ERROR (Day Log): Missing closing brace in record: {record}")
                 return

            clean_record = record[:end_index + 1]
            
            try:
                parts = clean_record.split("GUI_DAY_LOG_START:")[1].strip().split(',', 3)
                if len(parts) != 4:
                     raise ValueError(f"Malformed record structure. Expected 4 parts, got {len(parts)}.")
                
                policy = parts[0].strip().replace('\r', '').replace('\n', '') 

                sample_idx = int(parts[1].strip())
                day = int(parts[2].strip())
                
                json_part = parts[3].strip()
                start_json = json_part.find('{')
                end_json = json_part.rfind('}')
                
                if start_json == -1 or end_json == -1 or end_json < start_json:
                     raise ValueError(f"JSON payload not found or malformed in: {json_part}")
                
                json_string_only = json_part[start_json : end_json + 1]
                metrics = json.loads(json_string_only)

                if policy in self.policy_names and sample_idx >= self.sample_counts[policy]:
                    self.sample_counts[policy] = sample_idx + 1

                for metric, value in metrics.items():
                    if metric in METRICS:
                        # Data storage (fast operation) stays on the main thread
                        self.daily_data[policy][sample_idx][metric][day] = value 
                
                self.status_label.setText(f"Processing Policy: {policy}, Sample: {sample_idx}, Day: {day}")
                
                # CRITICAL: Emit signal to start the heavy plotting in the worker thread
                self.start_chart_update.emit(policy) 
            except Exception as e:
                print(f"CRITICAL PARSING ERROR (Day Log): {e} in record: {clean_record}")
        
        elif record.startswith("GUI_SUMMARY_LOG_START:"):
            # ... (Summary logic is unchanged) ...
            end_index = record.rfind('}')
            if end_index == -1:
                print(f"CRITICAL PARSING ERROR (Summary Log): Missing closing brace in record: {record}")
                return
                
            clean_record = record[:end_index + 1]
            
            try:
                json_part = clean_record.split("GUI_SUMMARY_LOG_START:")[1].strip()
                start_json = json_part.find('{')
                end_json = json_part.rfind('}')
                
                if start_json == -1 or end_json == -1 or end_json < start_json:
                     raise ValueError(f"JSON payload not found or malformed in: {json_part}")
                
                json_string_only = json_part[start_json : end_json + 1]
                summary_data = json.loads(json_string_only)
                
                self.summary_data = summary_data
                self.is_complete = True
                self.status_label.setText("Simulation Complete. Final results loaded.")
                self.redraw_summary_chart()
            except Exception as e:
                print(f"CRITICAL PARSING ERROR (Summary Log): {e} in record: {clean_record}")

    def redraw_summary_chart(self):
        """Redraws the summary chart."""
        if not self.summary_data:
            self.summary_ax.text(0.5, 0.5, "No summary data available.", 
                                 transform=self.summary_ax.transAxes, ha='center')
            return

        self.summary_ax.clear()
        
        log = self.summary_data['log']
        log_std = self.summary_data['log_std']
        
        metric_labels = METRICS
        
        policy_names = self.summary_data['policies']
        n_policies = len(policy_names)
        n_metrics = len(metric_labels)
        bar_width = 0.8 / n_policies
        
        x = np.arange(n_metrics)

        for i, policy in enumerate(policy_names):
            means = [log[policy][j] for j in range(n_metrics)]
            stds = [log_std[policy][j] for j in range(n_metrics)]
            
            r = x + bar_width * i
            
            self.summary_ax.bar(r, means, width=bar_width, 
                                edgecolor='grey', label=policy,
                                yerr=stds, capsize=5)

        self.summary_ax.set_xlabel("Metrics")
        self.summary_ax.set_ylabel("Mean Value")
        self.summary_ax.set_title("Average and Standard Deviation per Metric (Across Samples)")
        
        self.summary_ax.set_xticks(x + bar_width * (n_policies - 1) / 2, metric_labels)
        self.summary_ax.tick_params(axis='x', rotation=45)

        self.summary_ax.legend()
        self.summary_fig.canvas.draw_idle()
