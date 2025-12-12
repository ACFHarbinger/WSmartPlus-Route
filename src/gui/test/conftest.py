
import pytest
import sys
from unittest.mock import MagicMock
from PySide6.QtWidgets import QApplication, QWidget

# Mock dependencies that are hard to test or unnecessary for unit logic
# We do this at the top level to catch imports in the modules under test

# Mock FigureCanvasQTAgg to be a QWidget so layouts accept it
class MockCanvas(QWidget):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def setSizePolicy(self, *args): pass
    def updateGeometry(self): pass
    def draw_idle(self): pass

# Setup mocks *before* importing app modules
sys.modules['folium'] = MagicMock()
sys.modules['matplotlib.figure'] = MagicMock()

mock_backend = MagicMock()
mock_backend.FigureCanvasQTAgg = MockCanvas
sys.modules['matplotlib.backends.backend_qtagg'] = mock_backend
# Ensure direct import works too
sys.modules['matplotlib.backends.backend_qtagg'].FigureCanvasQTAgg = MockCanvas


from src.gui.windows.ts_results_window import SimulationResultsWindow

@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication exists for the session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app

@pytest.fixture
def results_window(qapp):
    """Fixture to create and tear down the SimulationResultsWindow."""
    win = SimulationResultsWindow(policy_names=['DefaultPolicy'])
    # Mock the redraw method to avoid Matplotlib interactions
    win.redraw_summary_chart = MagicMock()
    yield win
    win.close()
    win.deleteLater()
