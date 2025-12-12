
import pytest
import json
from unittest.mock import MagicMock
from src.gui.windows.ts_results_window import SimulationResultsWindow


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
    assert results_window.policy_names == ['DefaultPolicy']
    assert isinstance(results_window.summary_data, dict)
    assert results_window.summary_data['policies'] == []

def test_process_summary_merging(results_window):
    """Test merging of multiple GUI_SUMMARY_LOG_START records."""
    
    # Record 1
    summary1 = {
        "log": {"PolicyA": [10]},
        "log_std": {"PolicyA": [1]},
        "policies": ["PolicyA"],
        "n_samples": 5
    }
    line1 = "GUI_SUMMARY_LOG_START: " + json.dumps(summary1)
    
    # Record 2
    summary2 = {
        "log": {"PolicyB": [20]},
        "log_std": {"PolicyB": [2]},
        "policies": ["PolicyB"],
        "n_samples": 6
    }
    line2 = "GUI_SUMMARY_LOG_START: " + json.dumps(summary2)
    
    # Process
    results_window._process_single_record(line1)
    results_window._process_single_record(line2)
    
    data = results_window.summary_data
    
    # Assertions
    assert "PolicyA" in data['policies']
    assert "PolicyB" in data['policies']
    assert len(data['policies']) == 2
    
    assert data['log']['PolicyA'] == [10]
    assert data['log']['PolicyB'] == [20]
    
    # Check that n_samples was updated to the latest
    assert data['n_samples'] == 6
    
    # Ensure redraw was called
    assert results_window.redraw_summary_chart.call_count == 2
