from gui.src.tabs.evaluation.eval_input_output import EvalIOTab
from gui.src.tabs.evaluation.eval_problem import EvalProblemTab


def test_eval_problem_tab(qapp):
    tab = EvalProblemTab()
    # Check if get_params method exists and returns a dict
    assert hasattr(tab, "get_params")
    params = tab.get_params()
    assert isinstance(params, dict)
    tab.deleteLater()
    qapp.processEvents()


def test_eval_io_tab(qapp):
    tab = EvalIOTab()
    assert hasattr(tab, "get_params")
    params = tab.get_params()
    assert isinstance(params, dict)
    tab.deleteLater()
    qapp.processEvents()
