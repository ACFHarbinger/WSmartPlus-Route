from gui.src.tabs.hyperparam_optim import HyperParamOptimParserTab
from gui.src.tabs.meta_rl_train import MetaRLTrainParserTab
from gui.src.tabs.scripts import RunScriptsTab
from gui.src.tabs.ts_tab import TestSuiteTab


def test_hyperparam_optim_tab(qapp):
    tab = HyperParamOptimParserTab()
    assert hasattr(tab, "get_params")
    params = tab.get_params()
    assert "hop_method" in params
    assert "timeout" in params


def test_meta_rl_train_tab(qapp):
    tab = MetaRLTrainParserTab()
    assert hasattr(tab, "get_params")
    assert isinstance(tab.get_params(), dict)


def test_scripts_tab(qapp):
    tab = RunScriptsTab()
    # Check if this tab has params or just actions
    # Usually it's a runner, but let's check init
    assert tab is not None


def test_test_suite_tab(qapp):
    tab = TestSuiteTab()
    assert tab is not None
