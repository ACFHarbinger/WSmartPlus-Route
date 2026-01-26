import json
from unittest.mock import MagicMock, mock_open, patch

import pytest

from gui.src.tabs.analysis.output_analysis import OutputAnalysisTab


class TestOutputAnalysisInteractions:
    @pytest.fixture
    def output_tab(self, qapp):
        # Patch QTimer to avoid async delays in tests if used
        with patch("gui.src.tabs.analysis.output_analysis.QTimer"):
            tab = OutputAnalysisTab()
            tab.figure = MagicMock()
            tab.canvas = MagicMock()
            yield tab
            tab.shutdown()

    def test_load_files_json(self, output_tab):
        """Test loading a standard JSON output file."""
        mock_data = {"policy1_emp": {"profit": 100, "cost": 50}, "policy2_emp": {"profit": 200, "cost": 40}}

        # Mock file selection and opening
        with patch("PySide6.QtWidgets.QFileDialog.getOpenFileNames", return_value=(["/path/to/res.json"], "filter")):
            with patch("builtins.open", mock_open(read_data=json.dumps(mock_data))):
                with patch("os.path.basename", return_value="res.json"):
                    output_tab.load_files()

        # Verify data structure
        assert output_tab.json_data is not None
        assert "profit" in output_tab.json_data
        assert "cost" in output_tab.json_data
        assert len(output_tab.json_data["__Policy_Names__"]) == 2

        # Verify UI updates
        assert output_tab.y_key_combo.findText("profit") >= 0
        assert output_tab.plot_btn.isEnabled()
        assert "Loaded/Merged Files" in output_tab.text_view.toPlainText()

    def test_clear_data(self, output_tab):
        """Test clearing data resets state."""
        output_tab.json_data = {"some": "data"}
        output_tab.plot_btn.setEnabled(True)
        # Mock sim_windows to check closure
        mock_win = MagicMock()
        output_tab.sim_windows.append(mock_win)

        with patch("PySide6.QtWidgets.QMessageBox.information"):  # Suppress info box
            output_tab.clear_data()

        assert output_tab.json_data is None
        assert not output_tab.plot_btn.isEnabled()
        assert output_tab.y_key_combo.count() == 0
        assert output_tab.figure.clear.called
        assert mock_win.close.called
        assert len(output_tab.sim_windows) == 0

    def test_plot_metric_vs_policy(self, output_tab):
        """Test plotting a metric against policy names (Default view)."""
        output_tab.json_data = {
            "profit": [100, 200],
            "__Policy_Names__": ["p1", "p2"],
            "__Distributions__": ["emp", "emp"],
            "__File_IDs__": ["f1", "f1"],
        }

        # Setup selection
        output_tab.y_key_combo.addItem("profit")
        output_tab.y_key_combo.setCurrentText("profit")
        output_tab.x_key_combo.addItem("Policy Names")
        output_tab.x_key_combo.setCurrentText("Policy Names")
        output_tab.dist_combo.addItem("All")
        output_tab.dist_combo.setCurrentText("All")
        output_tab.chart_type_combo.setCurrentText("Line Chart")

        output_tab.plot_json_key()

        ax = output_tab.figure.add_subplot.return_value
        assert ax.plot.called
        assert ax.set_xlabel.call_args[0][0] == "Policy Name [Distribution]"
        assert output_tab.canvas.draw.called

    def test_plot_metric_vs_metric_pareto(self, output_tab):
        """Test plotting metric vs metric with Pareto front enabled."""
        # p2 dominates p1 (p2 has less cost AND more profit)
        output_tab.json_data = {
            "profit": [10, 20],
            "cost": [100, 50],
            "__Policy_Names__": ["p1", "p2"],
            "__Distributions__": ["emp", "emp"],
            "__File_IDs__": ["f1", "f1"],
        }

        output_tab.y_key_combo.addItem("profit")
        output_tab.y_key_combo.setCurrentText("profit")
        output_tab.x_key_combo.addItem("cost")  # Metric X
        output_tab.x_key_combo.setCurrentText("cost")
        output_tab.dist_combo.addItem("All")
        output_tab.dist_combo.setCurrentText("All")
        output_tab.pareto_check.setChecked(True)

        output_tab.plot_json_key()

        ax = output_tab.figure.add_subplot.return_value

        # Verify scatter plotting (calls plot with 'o' marker)
        # And check for "Pareto Front" in legend call if logic reached
        # Or check if ax.plot was called with label='Pareto Front'

        calls = ax.plot.call_args_list
        pareto_found = False
        for c in calls:
            if c[1].get("label") == "Pareto Front":
                pareto_found = True
                break

        assert pareto_found

    def test_show_plot_dialog_clearing(self, output_tab):
        """Test the dialog logic for clearing data after plotting."""
        output_tab.json_data = {"data": 1}

        # Patch QMessageBox where it is imported
        with patch("gui.src.tabs.analysis.output_analysis.QMessageBox") as MockBox:
            instance = MockBox.return_value

            # Create mock buttons
            btn_plot = MagicMock()
            btn_clear = MagicMock()

            # Side effect for addButton:
            # 1. Plot Button
            # 2. Clear Button
            # 3. Cancel Button
            instance.addButton.side_effect = [btn_plot, btn_clear, MagicMock()]

            # clickedButton returns clear button matches the second addButton call
            instance.clickedButton.return_value = btn_clear

            # Mock QTimer
            with patch("gui.src.tabs.analysis.output_analysis.QTimer.singleShot") as mock_timer:
                # Manually mock the method to ensure interception
                output_tab.plot_json_key = MagicMock()

                output_tab.show_plot_dialog()

                assert output_tab.plot_json_key.called
                assert mock_timer.called
                # Timer calls _clear_data_state_only
                assert mock_timer.call_args[0][1] == output_tab._clear_data_state_only
