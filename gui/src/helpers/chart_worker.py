from PySide6.QtCore import QMutex, QMutexLocker, QObject, Signal, Slot


class ChartWorker(QObject):
    """
    Worker that processes scalar data AND bin state history for plotting.
    It NEVER touches Matplotlib objects.
    """

    data_ready = Signal(str, dict)

    def __init__(
        self,
        daily_data,
        historical_bin_data,
        latest_bin_data,
        metrics_to_plot,
        data_mutex: QMutex,
        parent=None,
    ):
        super().__init__(parent)
        self.daily_data = daily_data
        self.historical_bin_data = historical_bin_data
        self.latest_bin_data = latest_bin_data
        self.metrics_to_plot = metrics_to_plot
        self.data_mutex = data_mutex

    @Slot(str)
    def process_data(self, target_key):
        """
        Reads scalar metrics AND bin state data, then emits them.
        """
        processed_data = {
            "max_days": 0,
            "metrics": {},
            # [NEW] Container for bin state info
            "bin_state": {},
        }

        # --- CRITICAL SECTION: DATA READING (Protected) ---
        with QMutexLocker(self.data_mutex):
            # 1. Process Scalar Metrics (Time Series)
            if target_key in self.daily_data:
                max_days = 0
                for metric in self.metrics_to_plot:
                    data_map = self.daily_data[target_key].get(metric, {})
                    if data_map:
                        max_days = max(max_days, max(data_map.keys()))

                processed_data["max_days"] = max_days

                for metric in self.metrics_to_plot:
                    data_map = self.daily_data[target_key].get(metric, {})
                    days = sorted(data_map.keys())
                    values = [data_map[d] for d in days]

                    processed_data["metrics"][metric] = {"days": days, "values": values}

            # 2. [NEW] Process Bin State Data (For labels/updates)
            # We don't need to deep copy the massive historical arrays here unless
            # visual tearing is an issue (usually not for this update rate).
            # We mostly need the latest scalars for the info label.
            if target_key in self.latest_bin_data:
                processed_data["bin_state"] = self.latest_bin_data[target_key].copy()

        # --- END CRITICAL SECTION ---

        self.data_ready.emit(target_key, processed_data)
