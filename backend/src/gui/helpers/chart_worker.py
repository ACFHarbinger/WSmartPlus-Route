import numpy as np

from PySide6.QtCore import QObject, Signal, Slot
try:
    from backend.src.utils.definitions import METRICS
except ImportError:
    METRICS = ['overflows', 'kg', 'ncol', 'km', 'kg/km', 'cost'] 


class ChartWorker(QObject):
    """
    Worker that modifies the Matplotlib figure data and signals the main thread to draw.
    """
    # Signal emitted when plotting operations are complete, carrying the canvas reference.
    draw_request = Signal(object) 
    
    def __init__(self, daily_data, sample_counts, policy_chart_widgets, colors, parent=None):
        super().__init__(parent)
        self.daily_data = daily_data
        self.sample_counts = sample_counts
        self.policy_chart_widgets = policy_chart_widgets
        self.colors = colors

    @Slot(str)
    def update_figure(self, target_policy):
        """
        Slot executed in the worker thread. Performs data processing and figure changes.
        """
        if target_policy not in self.policy_chart_widgets:
            return

        chart_data = self.policy_chart_widgets[target_policy]
        fig = chart_data['fig']
        axes = chart_data['axes']
        
        # --- Heavy Plotting Logic (Done in Worker Thread) ---
        max_days = 0 
        
        for sample_idx in range(self.sample_counts[target_policy]):
            for metric in METRICS:
                if self.daily_data[target_policy][sample_idx][metric]:
                    max_days = max(max_days, max(self.daily_data[target_policy][sample_idx][metric].keys()))
                    break

        for i, metric in enumerate(METRICS):
            ax = axes[i]
            # CLEARING and setting limits is generally safe in the worker if done carefully,
            # but updating the canvas must be signaled back to the main thread.
            ax.clear() 
            ax.grid(True)
            ax.set_title(f'{metric} for {target_policy}')

            all_values_for_metric = []
            plotted_any_line = False
            
            # 1. Plot all samples (Plotting happens here)
            for sample_idx in range(self.sample_counts[target_policy]):
                sample_color = self.colors[sample_idx % len(self.colors)]
                
                days = sorted(self.daily_data[target_policy][sample_idx][metric].keys())
                values = [self.daily_data[target_policy][sample_idx][metric][d] for d in days]
                
                if days:
                    all_values_for_metric.extend(values)
                    
                    ax.plot(days, values, label=f'Sample {sample_idx + 1}', 
                            marker='o', markersize=3, linestyle='-', color=sample_color)
                    plotted_any_line = True

            # 2. Set Dynamic Y-Axis Limits 
            if all_values_for_metric:
                min_val = min(all_values_for_metric)
                max_val = max(all_values_for_metric)
                y_range = max_val - min_val
                
                if y_range == 0:
                    ax.set_ylim(min_val - 1, max_val + 1)
                else:
                    buffer = y_range * 0.05
                    ax.set_ylim(min_val - buffer, max_val + buffer)
            else:
                ax.set_ylim(0, 1) 

            # 3. Set X-Axis Limits and Ticks
            if max_days > 0:
                ax.set_xlim(0.8, max_days + 0.2)
                ax.set_xticks(np.arange(1, max_days + 1, 1))
            else:
                ax.set_xlim(0, 100) 


            if i == len(METRICS) - 1:
                ax.set_xlabel("Day")
                
            if plotted_any_line and self.sample_counts[target_policy] > 1 and i == 0:
                ax.legend(loc='upper right', fontsize='small')
        
        # --- End Heavy Plotting Logic ---

        # Signal the main thread to perform the final draw operation
        self.draw_request.emit(chart_data['canvas'])
