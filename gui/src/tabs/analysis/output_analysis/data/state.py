class OutputDataState:
    def __init__(self):
        self.json_data = None
        self.sim_windows = []
        self._all_loaded_json_paths = []
        self.tb_process = None

    def clear(self):
        """Resets the data state."""
        self.json_data = None
        self._all_loaded_json_paths = []

        for win in self.sim_windows:
            if win is not None:
                win.close()
        self.sim_windows = []

        if self.tb_process:
            self.tb_process.terminate()
            self.tb_process = None

    def add_loaded_path(self, fpath):
        self._all_loaded_json_paths.append(fpath)

    def get_loaded_paths(self):
        return sorted(list(set(self._all_loaded_json_paths)))
