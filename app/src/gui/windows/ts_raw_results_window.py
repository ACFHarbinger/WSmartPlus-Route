import sys
from PySide6.QtWidgets import QWidget, QVBoxLayout, QTextEdit, QLabel
from PySide6.QtCore import Qt

class RawLogWindow(QWidget):
    """
    A temporary window to display all raw data received from the external process.
    Used for debugging the QProcess data pipe and parsing errors.
    """
    def __init__(self, policies):
        super().__init__()
        self.setWindowTitle("RAW Log Data Monitor")
        self.setMinimumSize(800, 600)
        self.setWindowFlags(self.windowFlags() | Qt.Window) 

        main_layout = QVBoxLayout(self)
        
        self.status_label = QLabel("Monitoring external process output...")
        self.status_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        main_layout.addWidget(self.status_label)
        
        self.log_area = QTextEdit()
        self.log_area.setReadOnly(True)
        self.log_area.setStyleSheet("font-family: monospace; font-size: 10pt; background: #2e3436; color: #d3d7cf;")
        main_layout.addWidget(self.log_area)

    def parse_buffer(self, buffer: str) -> str:
        """
        Instead of parsing, this method simply appends and prints the raw data.
        It assumes the buffer contains the *new* raw output.
        """
        # Append the new output to the text area
        self.log_area.append(buffer)
        
        # If the log area gets too big, clear it to prevent slowdowns
        if self.log_area.document().blockCount() > 500:
             self.log_area.clear()
             self.log_area.append("--- Log Cleared (Overflow Prevention) ---")
             
        # In this debug window, we don't worry about keeping a buffer
        # as we are simply validating reception.
        return "" # We assume all data consumed for display