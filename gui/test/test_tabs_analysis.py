
from unittest.mock import patch
from .conftest import MockDataLoadWorker
from gui.src.tabs.analysis.input_analysis import InputAnalysisTab
from gui.src.tabs.analysis.output_analysis import OutputAnalysisTab


# Mock DataLoadWorker where it is imported in input_analysis
@patch('gui.src.tabs.analysis.input_analysis.DataLoadWorker', new=MockDataLoadWorker)
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
