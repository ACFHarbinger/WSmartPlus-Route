
import pytest
from unittest.mock import MagicMock, patch
from gui.tabs.analysis.input_analysis import InputAnalysisTab
from gui.tabs.analysis.output_analysis import OutputAnalysisTab
from gui.test.conftest import MockDataLoadWorker

# Mock DataLoadWorker where it is imported in input_analysis
@patch('gui.tabs.analysis.input_analysis.DataLoadWorker', new=MockDataLoadWorker)
def test_input_analysis_tab_init(qapp):
    """Test initialization of InputAnalysisTab."""
    tab = InputAnalysisTab()
    assert tab.load_btn.text() == "Load Data File (CSV/XLSX/PKL)"
    # Clean up thread just in case
    tab.worker_thread.quit()
    tab.worker_thread.wait()

def test_output_analysis_tab_init(qapp):
    """Test initialization of OutputAnalysisTab."""
    tab = OutputAnalysisTab()
    assert hasattr(tab, 'load_btn') or hasattr(tab, 'plot_btn')
