from unittest.mock import patch

from gui.src.tabs.analysis.input_analysis import InputAnalysisTab
from gui.src.tabs.analysis.output_analysis import OutputAnalysisTab

from .conftest import MockDataLoadWorker


# Mock DataLoadWorker where it is imported in input_analysis
@patch("gui.src.tabs.analysis.input_analysis.DataLoadWorker", new=MockDataLoadWorker)
def test_input_analysis_tab_init(qapp):
    """Test initialization of InputAnalysisTab."""
    tab = InputAnalysisTab()
    assert tab.load_btn.text() == "Load Data File (CSV/XLSX/PKL)"
    # Clean up
    if tab.worker_thread.isRunning():
        tab.worker_thread.quit()
        tab.worker_thread.wait()
    tab.deleteLater()
    qapp.processEvents()


def test_output_analysis_tab_init(qapp):
    """Test initialization of OutputAnalysisTab."""
    tab = OutputAnalysisTab()
    # Buttons are in the controls widget
    assert hasattr(tab.controls, "load_btn")
    assert hasattr(tab.controls, "plot_btn")
    tab.deleteLater()
    qapp.processEvents()
