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
            return window

    def test_initialization(self, results_window):
        """Test initial state."""
        assert results_window.policy_names == ["p1"]
        assert results_window.status_label.text().startswith("Waiting")
        assert results_window.tabs.count() > 0

    def test_process_day_log(self, results_window):
        """Test processing of a day log line."""
        # Simulated log line
        log_data = {"overflows": 5, "profit": 100.0, "tour": {}}
        line = f"GUI_DAY_LOG_START: p1, 1, 0, {json.dumps(log_data)}"

        # Invoke processing directly (simulate signal)
        results_window._process_single_record(line)

        # Verify data storage
        key = "p1 sample 1"
        assert 1 in results_window.available_samples_dict["p1"]

        assert results_window.daily_data[key]["overflows"][0] == 5.0

        # Verify UI Dropdown update
        assert results_window.live_policy_combo.findText("p1") >= 0

    def test_process_summary_log(self, results_window):
        """Test processing of a summary log line."""
        summary = {"policies": ["p1"], "log": {"p1": [10.0, 5.0]}, "log_std": {"p1": [1.0, 0.5]}, "n_samples": 10}
        line = f"GUI_SUMMARY_LOG_START: {json.dumps(summary)}"

        # Patch redraw to avoid matplotlib errors in headless env
        with patch.object(results_window, "redraw_summary_chart") as mock_redraw:
            results_window._process_single_record(line)

            assert results_window.summary_data["n_samples"] == 10
            assert "p1" in results_window.summary_data["policies"]
            assert mock_redraw.called

    def test_chart_update(self, results_window):
        """Test chart update logic."""
        # Setup pre-reqs
        key = "p1 sample 1"
        results_window.live_policy_combo.addItem("p1")
        results_window.live_sample_combo.addItem("1")
        results_window.live_policy_combo.setCurrentText("p1")
        results_window.live_sample_combo.setCurrentText("1")

        processed_data = {"max_days": 10, "metrics": {"profit": {"days": [0, 1], "values": [100, 200]}}}

        # Mock canvas draw
        canvas_mock = results_window.live_ui_components["line_canvas"] = MagicMock()

        results_window._update_chart_on_main_thread(key, processed_data)

        assert canvas_mock.draw_idle.called

    def test_dropdown_logic(self, results_window):
        """Test dynamic dropdown updates."""
        results_window._update_live_dropdowns("p2", 2)

        assert results_window.live_policy_combo.findText("p2") >= 0

        # If we select p2, sample 2 should appear
        results_window.live_policy_combo.setCurrentText("p2")
        # Trigger change handler manually if needed, or rely on signal if connected
        # Signal is connected. But QTest or direct call?
        # Direct call to verify logic:
        # The logic _update_live_dropdowns adds sample if policy is current.
        # But here we just added it.
        # Let's test _on_live_policy_changed population

        results_window.available_samples_dict["p2"].add(2)
        results_window._on_live_policy_changed()

        assert results_window.live_sample_combo.findText("2") >= 0
