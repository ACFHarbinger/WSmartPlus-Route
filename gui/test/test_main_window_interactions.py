import sys
from unittest.mock import MagicMock, patch

import pytest

from gui.src.windows.main_window import MainWindow


class TestMainWindowInteractions:
    @pytest.fixture
    def main_window(self, qapp):
        # Patch DataLoadWorker globally for MainWindow initialization to avoid real threads
        with patch("gui.src.tabs.analysis.input_analysis.DataLoadWorker", new=MagicMock()):
            win = MainWindow(test_only=True)
            yield win
            win.close()
            # Ensure pending events (like cleanup) are processed
            qapp.processEvents()

    def test_command_change_updates_tabs(self, main_window):
        """Test that changing the command combo updates the visible tabs."""
        # Default is "Train Model"
        assert main_window.command_combo.currentText() == "Train Model"
        # Check first tab title
        assert main_window.tabs.tabText(0) == "Data"

        # Change to "Generate Data"
        main_window.command_combo.setCurrentText("Generate Data")

        # Verify tabs changed
        assert main_window.tabs.tabText(0) == "General Output"

        # Change to "Analysis" (Data Analysis)
        main_window.command_combo.setCurrentText("Data Analysis")
        assert main_window.tabs.tabText(0) == "Input Analysis"

    def test_run_command_simulation(self, main_window):
        """Test 'Run Command' in simulation mode."""
        main_window.preview.setPlainText("python main.py simulated")

        # Patch mediator safely to prevent side effects during run command
        with patch.object(main_window, "mediator", MagicMock()):
            with patch("PySide6.QtWidgets.QMessageBox.information") as mock_info:
                main_window.run_command()

                mock_info.assert_called()
                args = mock_info.call_args[0]
                assert "Command Simulation" in args[1]
                assert "python main.py simulated" in args[2]

    @patch("gui.src.windows.main_window.QProcess")
    def test_run_command_real_execution(self, MockProcess, main_window):
        """Test 'Run Command' launching a process."""
        # Reuse fixture, modify state
        main_window.test_only = False
        main_window.preview.setPlainText("echo hello")

        # Mock process instance
        process_instance = MockProcess.return_value
        process_instance.waitForStarted.return_value = True

        # Mock results window to avoid it popping up
        with patch("gui.src.windows.main_window.SimulationResultsWindow"):
            # Patch mediator safely
            with patch.object(main_window, "mediator", MagicMock()):
                main_window.run_command()

        assert process_instance.start.called
        call_args = process_instance.start.call_args
        program = call_args[0][0]
        args = call_args[0][1]

        if sys.platform.startswith("linux"):
            assert program == "sh"
            assert args == ["-c", "echo hello"]

    def test_toggle_theme(self, main_window):
        """Test theme toggling logic."""
        initial_theme = main_window.current_theme
        main_window.toggle_theme()
        assert main_window.current_theme != initial_theme

        main_window.toggle_theme()
        assert main_window.current_theme == initial_theme

    def test_close_cleanups(self, main_window):
        """Test that closing the window triggers shutdown on child tabs."""
        from PySide6.QtGui import QCloseEvent

        # Mock analysis tabs to verify shutdown calls
        mock_input = MagicMock()
        mock_output = MagicMock()

        main_window.analysis_tabs_map = {"Input Analysis": mock_input, "Output Analysis": mock_output}

        # Trigger close event with a real event object
        event = QCloseEvent()
        main_window.closeEvent(event)

        assert mock_input.shutdown.called
        assert mock_output.shutdown.called
