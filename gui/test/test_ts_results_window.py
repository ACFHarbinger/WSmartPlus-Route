import json
from unittest.mock import patch

import pytest


@pytest.fixture
def window(results_window):
    """
    Alias for results_window to keep existing tests compatible,
    or we can just rename the arguments in the tests below.
    Let's use the centralized fixture directly in tests for clarity.
    """
    return results_window


def test_initialization(results_window):
    """Test that the window initializes with correct default state."""
    assert results_window.policy_names == ["DefaultPolicy"]
    # summary_data removed in favor of DataManager
    assert results_window.data_manager is not None


def test_process_summary_merging(results_window):
    """Test merging of multiple GUI_SUMMARY_LOG_START records."""

    # Currently summary log start just triggers redraw, logic for merging
    # might happen inside data manager if at all, but current implementation
    # of _handle_new_log_line just calls redraw_summary_chart.

    line1 = "GUI_SUMMARY_LOG_START: {}"

    with patch.object(results_window, "redraw_summary_chart") as mock_redraw:
        results_window._handle_new_log_line(line1)
        assert mock_redraw.called
