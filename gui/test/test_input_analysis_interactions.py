from unittest.mock import MagicMock, patch

import pytest

from gui.src.tabs.analysis.input_analysis import InputAnalysisTab

from .conftest import MockDataLoadWorker


class TestInputAnalysisInteractions:
    @pytest.fixture
    def input_tab(self, qapp):
        # Patch DataLoadWorker to avoid real thread creation
        with patch("gui.src.tabs.analysis.input_analysis.DataLoadWorker", new=MockDataLoadWorker):
            tab = InputAnalysisTab()
            # Mock figure and canvas to verify plotting calls without rendering
            tab.figure = MagicMock()
            tab.canvas = MagicMock()
            yield tab
            # Cleanup
            if tab.worker_thread.isRunning():
                tab.worker_thread.quit()
                tab.worker_thread.wait()
            tab.close()
            tab.deleteLater()
            qapp.processEvents()

    def test_handle_successful_load(self, input_tab):
        """Test that loading data populates the slice selector and dataframes."""
        data = {"col1": [1, 2], "col2": [3, 4]}
        thread_safe_list = [("Slice1", data)]

        input_tab._handle_successful_load(thread_safe_list)

        # Verify selector population
        assert input_tab.slice_selector.count() == 1
        assert "Slice1" in input_tab.slice_selector.currentText()
        assert "Slice1 (2x2)" in input_tab.dfs

        # Verify buttons enabled
        assert input_tab.load_btn.isEnabled()
        assert input_tab.plot_btn.isEnabled()

        # Verify default selection triggers switch
        assert input_tab.current_slice_key == input_tab.slice_selector.currentText()
        assert input_tab.x_axis_combo.count() == 2

    def test_switch_current_df(self, input_tab):
        """Test that switching the selected slice updates columns and views."""
        data1 = {"A": [1], "B": [2]}
        data2 = {"X": [10], "Y": [20]}
        # Load two slices
        input_tab._handle_successful_load([("S1", data1), ("S2", data2)])

        # Initially S1 should be selected (index 0)
        assert "S1" in input_tab.current_slice_key
        assert input_tab.x_axis_combo.itemText(0) == "A"

        # Switch to S2 (Index 1)
        input_tab.slice_selector.setCurrentIndex(1)
        # Note: In programmatic changes, ensure the slot is called or call manually.
        # currentIndexChanged usually fires on programmatic change in PySide6 depending on settings,
        # but calling manually ensures test robustness if signal ownership is complex.
        # InputAnalysisTab connects currentIndexChanged to _switch_current_df.
        # Let's verify if we need to call it manually. For now assume signal works or call it.
        # We'll explicitly call it to strict unit test the logic.
        input_tab._switch_current_df()

        assert "S2" in input_tab.current_slice_key
        assert input_tab.x_axis_combo.itemText(0) == "X"

    def test_update_chart_controls(self, input_tab):
        """Test enabling/disabling axes combos based on chart type."""
        # Select Heatmap
        input_tab.chart_type_combo.setCurrentText("Heatmap")
        input_tab._update_chart_controls()  # Slot call

        assert not input_tab.x_axis_combo.isEnabled()
        assert not input_tab.y_axis_combo.isEnabled()

        # Select Line Chart
        input_tab.chart_type_combo.setCurrentText("Line Chart")
        input_tab._update_chart_controls()

        assert input_tab.x_axis_combo.isEnabled()
        assert input_tab.y_axis_combo.isEnabled()

    def test_plot_data_line_chart(self, input_tab):
        """Test plotting logic for Line Chart."""
        # Setup data
        data = {"Time": [1, 2, 3], "Value": [10, 20, 15]}
        input_tab._handle_successful_load([("Data", data)])

        # Select Axes
        input_tab.x_axis_combo.setCurrentText("Time")
        input_tab.y_axis_combo.setCurrentText("Value")
        input_tab.chart_type_combo.setCurrentText("Line Chart")

        # Call plot
        input_tab.plot_data()

        # Verify figure usage
        assert input_tab.figure.clear.called
        assert input_tab.figure.add_subplot.called
        ax = input_tab.figure.add_subplot.return_value
        assert ax.plot.called
        assert ax.set_xlabel.call_args[0][0] == "Time"

        # Verify canvas draw
        assert input_tab.canvas.draw.called

    def test_plot_data_validation_error(self, input_tab):
        """Test validation when axes are not selected."""
        input_tab._handle_successful_load([("Data", {"x": [1]})])
        input_tab.chart_type_combo.setCurrentText("Line Chart")

        # Deselect axes (or simulate empty selection if possible)
        # We'll just force combos to be empty or effectively empty text
        input_tab.x_axis_combo.setCurrentIndex(-1)
        input_tab.y_axis_combo.setCurrentIndex(-1)

        with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_msg:
            input_tab.plot_data()
            mock_msg.assert_called_with(input_tab, "Plot Error", "Please select both X and Y axes for this chart type.")

    def test_plot_data_heatmap(self, input_tab):
        """Test plotting logic for Heatmap."""
        data = {"A": [1, 2], "B": [3, 4]}
        input_tab._handle_successful_load([("Matrix", data)])
        input_tab.chart_type_combo.setCurrentText("Heatmap")

        input_tab.plot_data()

        ax = input_tab.figure.add_subplot.return_value
        assert ax.imshow.called
        assert input_tab.figure.colorbar.called
