import json
from unittest.mock import MagicMock, patch

import pytest

from gui.src.windows.ts_results_window import SimulationResultsWindow


class TestSimResultsWindowInteractions:
    @pytest.fixture
    def results_window(self, qapp):
        # Patch workers to prevent real thread startup
        with patch("gui.src.windows.ts_results_window.FileTailerWorker"), patch(
            "gui.src.windows.ts_results_window.ChartWorker"
        ), patch("PySide6.QtCore.QThread.start"):
            window = SimulationResultsWindow(policy_names=["p1"])
            yield window
            window.close()
            window.deleteLater()
            qapp.processEvents()

    def test_initialization(self, results_window):
        """Test initial state."""
        assert results_window.policy_names == ["p1"]
        assert results_window.status_label.text().startswith("Waiting")
        assert results_window.tabs.count() > 0

    def test_process_day_log(self, results_window):
        """Test processing of a day log line."""
        # Simulated log line
        log_data = {"overflows": 5, "profit": 100.0, "tour": {}}
        line = f"GUI_DAY_LOG_START:p1, sample, 0, {json.dumps(log_data)}"

        # Invoke processing via the handler
        results_window._handle_new_log_line(line)

        # Verify data storage in data_manager
        # data_manager.process_record extracts policy from the record if present,
        # otherwise logic depends on parse_log_line.
        # But wait, results_window._handle_new_log_line parses using split on "GUI_DAY_LOG_START:"
        # AND THEN calls parse_log_line.
        # data_manager.parse_log_line expects just JSON.
        # The line format in code: "GUI_DAY_LOG_START:" + json_str.
        # So my line above is correct if I stick to that.
        # However, data_manager.process_record requires "policy" and "sample" in the record.
        # My log_data above lacks them.
        # Let's add them.
        log_data = {"policy": "p1", "sample": "1", "day": 0, "overflows": 5, "profit": 100.0, "routes": {}}
        line = f"GUI_DAY_LOG_START:{json.dumps(log_data)}"

        results_window._handle_new_log_line(line)

        # Verify data storage
        key = "p1_1"
        assert "1" in results_window.data_manager.policy_samples["p1"]
        assert results_window.data_manager.day_data[key][0]["metrics"]["overflows"] == 5

        # Verify UI Dropdown update
        # Dashboard tab might store items. Check dashboard_tab directly.
        assert results_window.dashboard_tab.policy_combo.findText("p1") >= 0

    def test_process_summary_log(self, results_window):
        """Test processing of a summary log line."""
        # Summary log just triggers a redraw in current implementation
        line = "GUI_SUMMARY_LOG_START: {}"

        # Patch redraw to avoid matplotlib errors in headless env
        with patch.object(results_window, "redraw_summary_chart") as mock_redraw:
            results_window._handle_new_log_line(line)
            assert mock_redraw.called

    def test_chart_update(self, results_window):
        """Test chart update logic."""
        # Setup pre-reqs
        key = "p1_1"
        results_window.dashboard_tab.policy_combo.addItem("p1")
        results_window.dashboard_tab.sample_combo.addItem("1")
        results_window.dashboard_tab.policy_combo.setCurrentText("p1")
        results_window.dashboard_tab.sample_combo.setCurrentText("1")

        processed_data = {"max_days": 10, "metrics": {"profit": {"days": [0, 1], "values": [100, 200]}}}

        # Mock canvas draw
        # dashboard_tab has the canvas logic? No, check if live_ui_components exists.
        # Use dashboard_tab objects if possible or skip if too complex to mock deep structure.
        # checking results_window.dashboard_tab...
        # Assuming it has a method to update.
        # But _update_ui_on_data_ready delegates to tabs.
        # Let's just call _update_ui_on_data_ready and verify it tries to update dashboard tab

        with patch.object(results_window.dashboard_tab, "day_combo") as mock_combo:
             results_window._update_ui_on_data_ready(key, processed_data)
             mock_combo.clear.assert_called()

    def test_dropdown_logic(self, results_window):
        """Test dynamic dropdown updates."""
        # Current logic: _handle_new_log_line -> data_manager.process_record -> dashboard_tab.update_samples

        # We simulate the dashboard tab update method
        with patch.object(results_window.dashboard_tab, "update_samples") as mock_update:
            # Create a fake record that triggers update
            log_data = {"policy": "p2", "sample": "2", "day": 0}
            line = f"GUI_DAY_LOG_START:{json.dumps(log_data)}"

            results_window._handle_new_log_line(line)

            mock_update.assert_called()
            # args = mock_update.call_args[0][0]
            # assert "2" in args (depending on implementation detail of update_samples)
