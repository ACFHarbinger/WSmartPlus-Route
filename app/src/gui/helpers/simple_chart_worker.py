# --- File: simple_chart_worker.py (Per-Sample Tab Fix) ---
import numpy as np

from PySide6.QtCore import QObject, Signal, Slot, QMutex, QMutexLocker


class SimpleChartWorker(QObject):
    """
    Worker that plots data for a *single* (Policy, Sample) key.
    """
    draw_request = Signal(object) 
    
    def __init__(self, daily_data, policy_chart_widgets, color, metrics_to_plot, 
                 data_mutex: QMutex, parent=None):
        super().__init__(parent)
        self.daily_data = daily_data
        self.policy_chart_widgets = policy_chart_widgets
        self.color = color # Single color, not a list
        self.metrics_to_plot = metrics_to_plot
        self.data_mutex = data_mutex 

    @Slot(str)
    def update_figure(self, target_key):
        """
        Slot executed in the worker thread. 
        target_key is the unique (Policy, Sample) string.
        """
        if target_key not in self.policy_chart_widgets:
            return

        chart_data = self.policy_chart_widgets[target_key]
        fig = chart_data['fig']
        axes = chart_data['axes']
        canvas = chart_data['canvas']

        # --- CRITICAL SECTION: DATA READING (Protected) ---
        with QMutexLocker(self.data_mutex):
            
            max_days = 0 
            
            # 1. Find max days for this specific key
            for metric in self.metrics_to_plot:
                if target_key in self.daily_data and \
                   metric in self.daily_data[target_key] and \
                   self.daily_data[target_key][metric]:
                    
                    max_days = max(max_days, max(self.daily_data[target_key][metric].keys()))
                    break 

            # 2. Iterate and plot metrics
            for i, metric in enumerate(self.metrics_to_plot):
                ax = axes[i]
                ax.clear() 
                ax.grid(True)
                ax.set_title(f'{metric} for {target_key}')
                ax.set_ylabel(metric)

                all_values_for_metric = []
                
                # Check if data exists for this metric
                if metric not in self.daily_data[target_key]:
                    continue

                # Get data for this single sample
                days = sorted(self.daily_data[target_key][metric].keys())
                values = [self.daily_data[target_key][metric][d] for d in days]
                
                if days:
                    all_values_for_metric.extend(values)
                    
                    # No loop, no legend, just plot the single line
                    ax.plot(days, values, 
                            marker='o', markersize=3, linestyle='-', color=self.color)

                # 3. Set Dynamic Y-Axis Limits
                if all_values_for_metric:
                    min_val = min(all_values_for_metric)
                    max_val = max(all_values_for_metric)
                    y_range = max_val - min_val
                    
                    if y_range == 0:
                        if max_val == 0:
                            ax.set_ylim(-0.1, 1.1)
                        else:
                            buffer = max_val * 0.1
                            ax.set_ylim(max(0, min_val - buffer), max_val + buffer)
                    else:
                        buffer = y_range * 0.05
                        ax.set_ylim(max(0, min_val - buffer), max_val + buffer)
                else:
                    ax.set_ylim(-0.1, 1.1) 

                # 4. Set X-Axis Limits and Ticks
                if max_days > 0:
                    ax.set_xlim(0.8, max_days + 0.2)
                    step = max(1, int(max_days / 10))
                    ax.set_xticks(np.arange(1, max_days + 1, step))
                else:
                    ax.set_xlim(0, 100) 

                if i == len(self.metrics_to_plot) - 1:
                    ax.set_xlabel("Day")
                
                # No legend needed for single-sample plots

        # --- END CRITICAL SECTION ---
        
        # Signal the main thread to perform the final draw operation
        self.draw_request.emit(canvas)