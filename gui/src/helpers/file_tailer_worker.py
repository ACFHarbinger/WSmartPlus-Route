"""
Worker for tailing log files in real-time.
"""

import os
import time

from PySide6.QtCore import QMutex, QObject, Signal, Slot


class FileTailerWorker(QObject):
    """
    Background worker that monitors a file and emits new lines as they are written.
    Useful for real-time log monitoring of external simulation processes.
    """

    # Signal emitted when a new complete log entry is found
    log_line_ready = Signal(str)

    def __init__(self, data_mutex: QMutex, log_path, parent=None):
        """
        Initialize FileTailerWorker.

        Args:
            data_mutex: Shared mutex for thread-safe access.
            log_path: Path to the log file to monitor.
            parent: Parent QObject.
        """
        super().__init__(parent)
        self.file_path = log_path
        self.data_mutex = data_mutex
        self._is_running = True

    def stop(self):
        """Method to safely stop the worker from the main thread."""
        self._is_running = False

    @Slot()
    def tail_file(self):
        """Continuously monitors the log file for new entries."""

        # Give the system time to start the simulation process and create the file
        time.sleep(1)

        current_pos = 0
        while self._is_running:
            try:
                # Wait for the file to exist
                if not os.path.exists(self.file_path):
                    time.sleep(0.5)
                    continue

                with open(self.file_path, "r") as f:
                    # Move to the last read position
                    f.seek(current_pos)

                    new_line = f.readline()

                    if new_line:
                        # Process the new line and emit the signal
                        self.log_line_ready.emit(new_line)

                        # Update the position to the end of the new line
                        current_pos = f.tell()
                    else:
                        # No new data, wait a bit before checking again
                        time.sleep(0.1)

            except Exception as e:
                # Handle file access errors gracefully
                print(f"File Tailing Error: {e}")
                time.sleep(1)
