from PySide6.QtCore import QObject, Signal, Slot, QMutex, QMutexLocker


class ChartWorker(QObject):
    """
    Worker that ONLY processes data for a (Policy, Sample) key and emits
    the results. It NEVER touches Matplotlib objects.
    """
    # Signal emits the target_key and a dictionary of processed data
    data_ready = Signal(str, dict) 
    
    def __init__(self, daily_data, metrics_to_plot, 
                 data_mutex: QMutex, parent=None):
        super().__init__(parent)
        self.daily_data = daily_data
        self.metrics_to_plot = metrics_to_plot
        self.data_mutex = data_mutex 

    @Slot(str)
    def process_data(self, target_key):
        """
        Slot executed in the worker thread. 
        Reads data and emits it for the main thread to plot.
        """
        
        processed_data = {
            'max_days': 0,
            'metrics': {}
        }
        
        # --- CRITICAL SECTION: DATA READING (Protected) ---
        with QMutexLocker(self.data_mutex):
            
            if target_key not in self.daily_data:
                # Emit empty data so the main thread can still clear the plot if needed
                self.data_ready.emit(target_key, processed_data)
                return

            max_days = 0 
            
            # 1. Find max days for this specific key
            for metric in self.metrics_to_plot:
                if metric in self.daily_data[target_key] and self.daily_data[target_key][metric]:
                    max_days = max(max_days, max(self.daily_data[target_key][metric].keys()))
            
            processed_data['max_days'] = max_days

            # 2. Iterate and process metrics
            for metric in self.metrics_to_plot:
                if metric not in self.daily_data[target_key]:
                    processed_data['metrics'][metric] = {'days': [], 'values': []}
                    continue

                # Get data for this single sample
                days = sorted(self.daily_data[target_key][metric].keys())
                values = [self.daily_data[target_key][metric][d] for d in days]
                
                processed_data['metrics'][metric] = {
                    'days': days,
                    'values': values
                }
                
        # --- END CRITICAL SECTION ---
        
        # Signal the main thread with the processed data
        self.data_ready.emit(target_key, processed_data)
