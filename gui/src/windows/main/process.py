"""
Process management logic for MainWindow.
"""

import sys

from PySide6.QtCore import QObject, QProcess, Signal
from PySide6.QtWidgets import QMessageBox

from ..ts_results_window import SimulationResultsWindow


class ProcessManager(QObject):
    """
    Handles QProcess execution and output redirection.
    """

    finished = Signal(int, QProcess.ExitStatus)
    output_received = Signal(str)

    def __init__(self, main_window):
        """
        Initialize the process manager.

        Args:
            main_window: The parent MainWindow instance.
        """
        super().__init__()
        self.window = main_window
        self.process = None
        self.output_buffer = ""
        self.results_window = None

    def run_command(self, command_str, main_command, test_only):
        """Starts the external command using QProcess."""
        if main_command == "Analysis":
            QMessageBox.information(self.window, "Info", "Use the buttons inside the Analysis tabs to load files.")
            return

        shell_command = command_str.replace(" \\\n  ", " ")

        if test_only:
            QMessageBox.information(
                self.window,
                "Command Simulation",
                f"The following command would be executed:\n\n{command_str}\n\n"
                "(Execution is simulated in this environment).",
            )
            self.finished.emit(0, QProcess.ExitStatus.NormalExit)
            return

        is_simulation = main_command == "Test Simulator"

        # Close existing results window
        if self.results_window and self.results_window.isVisible():
            self.results_window.close()
            self.results_window = None

        if is_simulation:
            test_sim_tab = self.window.test_sim_tabs_map["Simulator Settings"]
            policy_names = ["Unknown Policy"]
            if hasattr(test_sim_tab, "get_params"):
                policies_str = test_sim_tab.get_params().get("policies", "")
                policy_names = policies_str.split() if policies_str else ["Unknown Policy"]

            self.results_window = SimulationResultsWindow(policy_names)
            self.results_window.show()
        else:
            self.results_window = None

        if self.process is not None:
            self.process.terminate()
            self.process.waitForFinished(100)

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)

        self.process.readyReadStandardOutput.connect(self.read_output)
        self.process.finished.connect(self._on_finished)

        program = "sh" if sys.platform.startswith("linux") or sys.platform.startswith("darwin") else "cmd"
        if program == "sh":
            arguments = ["-c", shell_command]
        elif program == "cmd":
            arguments = ["/C", shell_command]
        else:
            parts = shell_command.split()
            program = parts[0]
            arguments = parts[1:]

        print(f"Starting process: {program} {' '.join(arguments)}")
        self.process.start(program, arguments)

        if not self.process.waitForStarted(200):
            error_msg = self.process.errorString()
            QMessageBox.critical(self.window, "Error", f"Failed to start external process: {error_msg}")
            self._on_finished(1, QProcess.ExitStatus.CrashExit)

    def read_output(self):
        """Reads output and feeds it to the results window."""
        if self.process is None:
            return
        output_bytes = self.process.readAllStandardOutput()
        output = output_bytes.data().decode()

        self.output_buffer += output

        if self.results_window:
            self.output_buffer = self.results_window.parse_buffer(self.output_buffer)

        non_structural_output = [line for line in output.splitlines() if not line.startswith("GUI_")]
        if non_structural_output:
            print("\n".join(non_structural_output))
            sys.stdout.flush()

    def _on_finished(self, exit_code, exit_status):
        """Handle process termination."""
        if exit_status == QProcess.ExitStatus.NormalExit and exit_code == 0:
            if self.results_window:
                self.results_window.status_label.setText("Simulation Complete: Success")
        else:
            if self.results_window:
                self.results_window.status_label.setText(f"Simulation Failed (Code: {exit_code})")

        self.process = None
        self.finished.emit(exit_code, exit_status)

    def cleanup(self):
        """Cleanup process and windows."""
        if self.results_window and self.results_window.isVisible():
            self.results_window.close()

        if self.process is not None and self.process.state() == QProcess.ProcessState.Running:
            self.process.terminate()
            self.process.waitForFinished(1000)
