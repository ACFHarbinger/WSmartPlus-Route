import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication, QWidget

# The project root is THREE levels up from conftest.py:
# conftest.py -> test -> gui -> WSmart-Route (Project Root)
project_root = Path(__file__).resolve().parent.parent.parent

# Add the project root to sys.path. This allows 'import gui.src...'
# to resolve 'gui' as a package within WSmart-Route/.
sys.path.insert(0, str(project_root))

# Mock dependencies that are hard to test or unnecessary for unit logic
# We do this at the top level to catch imports in the modules under test


# Mock FigureCanvasQTAgg to be a QWidget so layouts accept it
class MockCanvas(QWidget):
    def __init__(self, *args, **kwargs):
        parent = kwargs.get("parent")
        super().__init__(parent)

    def setSizePolicy(self, *args):
        pass

    def updateGeometry(self):
        pass

    def draw_idle(self):
        pass


# Mocking behavior
mock_backend = MagicMock()
mock_backend.FigureCanvasQTAgg = MockCanvas
sys.modules["matplotlib.backends.backend_qtagg"] = mock_backend
# Ensure direct import works too
sys.modules["matplotlib.backends.backend_qtagg"].FigureCanvasQTAgg = MockCanvas


# Mock DataLoadWorker to prevent thread issues in InputAnalysisTab
class MockDataLoadWorker(MagicMock):
    def moveToThread(self, thread):
        pass


# We can't easily sys.modules mock only one class from a module if the module has other stuff we need,
# but we can patch it in fixtures or assume the tests will patch it.
# However, let's try to mock the specific import if possible or just provide a fixture.
# A better approach for QThread/Worker components in UI tests is often to patch them during test setup.

from gui.src.windows.ts_results_window import SimulationResultsWindow  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """Ensure a QApplication exists for the session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    yield app
    # Process pending events like deleteLater()
    app.processEvents()


@pytest.fixture
def results_window(qapp):
    """Fixture to create and tear down the SimulationResultsWindow."""
    win = SimulationResultsWindow(policy_names=["DefaultPolicy"])
    # Mock the redraw method to avoid Matplotlib interactions
    win.redraw_summary_chart = MagicMock()
    yield win
    win.close()
    win.deleteLater()
