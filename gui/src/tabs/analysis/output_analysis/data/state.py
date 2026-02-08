"""Shared state management for output analysis plotting."""


class OutputDataState:
    """
    Holds the transient state of the output analysis tab, including loaded data and active processes.
    """

    def __init__(self):
        """
        Initialize the state with empty data and lists.
        """
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
        """
        Track a file path that has been loaded into the state.

        Args:
            fpath (str): Path to the loaded file.
        """
        self._all_loaded_json_paths.append(fpath)

    def get_loaded_paths(self):
        """
        Retrieve a sorted list of unique file paths currently loaded.

        Returns:
            list: Sorted list of unique file paths.
        """
        return sorted(list(set(self._all_loaded_json_paths)))
