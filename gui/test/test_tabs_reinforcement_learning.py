
import pytest
from gui.src.tabs.reinforcement_learning.rl_training import RLTrainingTab
from gui.src.tabs.reinforcement_learning.rl_model import RLModelTab
from gui.src.tabs.reinforcement_learning.rl_data import RLDataTab

def test_rl_training_tab(qapp):
    tab = RLTrainingTab()
    assert hasattr(tab, 'get_params')
    params = tab.get_params()
    assert 'n_epochs' in params
    assert 'lr_model' in params
    # Test custom components logic like toggle
    assert hasattr(tab, 'load_model_optim_header_widget')
    assert not tab.is_load_visible
    # Trigger toggle
    tab._toggle_load_model_optim()
    assert tab.is_load_visible

def test_rl_model_tab(qapp):
    tab = RLModelTab()
    assert hasattr(tab, 'get_params')
    assert isinstance(tab.get_params(), dict)

def test_rl_data_tab(qapp):
    tab = RLDataTab()
    assert hasattr(tab, 'get_params')
    assert isinstance(tab.get_params(), dict)
