
import pytest
from src.gui.tabs.generate_data.gd_problem import GenDataProblemTab
from src.gui.tabs.generate_data.gd_general import GenDataGeneralTab
from src.gui.tabs.generate_data.gd_advanced import GenDataAdvancedTab

def test_gd_problem_tab(qapp):
    tab = GenDataProblemTab()
    assert hasattr(tab, 'get_params')
    assert isinstance(tab.get_params(), dict)

def test_gd_general_tab(qapp):
    tab = GenDataGeneralTab()
    assert hasattr(tab, 'get_params')
    assert isinstance(tab.get_params(), dict)

def test_gd_advanced_tab(qapp):
    tab = GenDataAdvancedTab()
    assert hasattr(tab, 'get_params')
    assert isinstance(tab.get_params(), dict)
