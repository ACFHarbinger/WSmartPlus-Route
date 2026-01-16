from unittest.mock import MagicMock

from gui.src.tabs.test_simulator.ts_input_output import TestSimIOTab
from gui.src.tabs.test_simulator.ts_policy_parameters import TestSimPolicyParamsTab
from gui.src.tabs.test_simulator.ts_settings import TestSimSettingsTab


def test_ts_settings_tab(qapp):
    tab = TestSimSettingsTab()
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)


def test_ts_io_tab(qapp):
    mock_settings = MagicMock()
    tab = TestSimIOTab(settings_tab=mock_settings)
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)


def test_ts_policy_params_tab(qapp):
    tab = TestSimPolicyParamsTab()
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)
