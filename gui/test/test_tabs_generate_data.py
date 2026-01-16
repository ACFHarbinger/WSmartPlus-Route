from gui.src.tabs.generate_data.gd_advanced import GenDataAdvancedTab
from gui.src.tabs.generate_data.gd_general import GenDataGeneralTab
from gui.src.tabs.generate_data.gd_problem import GenDataProblemTab


def test_gd_problem_tab(qapp):
    tab = GenDataProblemTab()
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)


def test_gd_general_tab(qapp):
    tab = GenDataGeneralTab()
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)


def test_gd_advanced_tab(qapp):
    tab = GenDataAdvancedTab()
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)
