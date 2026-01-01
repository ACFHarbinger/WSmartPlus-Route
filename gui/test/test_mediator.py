
import pytest
from PySide6.QtWidgets import QWidget, QApplication
from PySide6.QtCore import Signal
from gui.src.core.mediator import UIMediator

# Mock Tab with signal
class MockTab(QWidget):
    paramsChanged = Signal()
    
    def __init__(self, params=None):
        super().__init__()
        self.params = params or {}
        
    def get_params(self):
        return self.params
        
    def update_param(self, key, value):
        self.params[key] = value
        self.paramsChanged.emit()

# Mock MainWindow
class MockMainWindow:
    def __init__(self):
        self.tabs = self

    # Mocking QTabWidget behavior roughly
    def tabText(self, index):
        return "Model" # Default for test

    def currentIndex(self):
        return 0

@pytest.fixture(scope="session")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    yield app

@pytest.fixture
def mediator(qapp):
    mw = MockMainWindow()
    return UIMediator(mw)

def test_register_tab_and_update(mediator):
    tab = MockTab({'epochs': 100})
    mediator.register_tab('Train Model', 'Model', tab)
    
    # Assert tab is registered
    assert mediator.tabs['Train Model']['Model'] == tab
    
    # Check if signal connects to update_preview (we can't easily check connection in PySide6 without spying)
    # But we can check if emitting signal triggers command_updated
    
    received_commands = []
    mediator.command_updated.connect(lambda cmd: received_commands.append(cmd))
    
    mediator.set_current_command('Train Model')
    # Initial update
    assert len(received_commands) == 1
    assert "--epochs 100" in received_commands[0]
    
    # Update param and emit
    tab.update_param('epochs', 200)
    # Should trigger update
    assert len(received_commands) == 2
    assert "--epochs 200" in received_commands[1]

def test_command_mapping(mediator):
    cmd = mediator.get_actual_command('Train Model')
    assert cmd == 'train'
    
    cmd = mediator.get_actual_command('Test Simulator')
    assert cmd == 'test_sim'
    
    # Test specific tab override logic (requires mocking MainWindow state better if we want to test 'hp_optim' etc)

def test_analysis_command(mediator):
    received_commands = []
    mediator.command_updated.connect(lambda cmd: received_commands.append(cmd))
    
    mediator.set_current_command('Analysis')
    assert "Analysis tools run directly" in received_commands[0]
