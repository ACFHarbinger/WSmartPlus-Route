import sys
from unittest.mock import MagicMock

from logic.src.ui.components.policy_viz import render_policy_viz


# Set up mock streamlit
def mock_cache_decorator(*args, **kwargs):
    def decorator(func):
        return func

    return decorator


if "streamlit" not in sys.modules:
    mock_st = MagicMock()
    mock_st.cache_data = mock_cache_decorator
    sys.modules["streamlit"] = mock_st
else:
    mock_st = sys.modules["streamlit"]
    if isinstance(mock_st, MagicMock):
        mock_st.cache_data = mock_cache_decorator


# Dynamically return appropriate number of columns
def mock_columns(num_or_spec):
    num = num_or_spec if isinstance(num_or_spec, int) else len(num_or_spec)
    return [MagicMock() for _ in range(num)]


mock_st.columns = mock_columns


def test_render_policy_viz_empty():
    mock_st.reset_mock()
    render_policy_viz({})
    mock_st.info.assert_called_once()


def test_render_policy_viz_alns():
    mock_st.reset_mock()
    viz_data = {
        "iteration": [0, 1],
        "best_cost": [100.0, 95.0],
        "current_cost": [110.0, 98.0],
        "temperature": [10.0, 9.0],
        "d_idx": [0, 1],
        "r_idx": [0, 1],
        "n_accepted": [1, 0],
        "n_improved": [1, 1],
    }
    render_policy_viz(viz_data, title="ALNS Test", smooth_window=2)
    mock_st.subheader.assert_called_with("ALNS Test")
    mock_st.plotly_chart.assert_called()


def test_render_policy_viz_hgs():
    mock_st.reset_mock()
    viz_data = {
        "generation": [0, 1],
        "best_cost": [100.0, 95.0],
        "mean_cost": [110.0, 105.0],
        "worst_cost": [120.0, 115.0],
        "no_improv": [0, 1],
        "restarted": [False, True],
    }
    render_policy_viz(viz_data)
    mock_st.plotly_chart.assert_called()


def test_render_policy_viz_aco():
    mock_st.reset_mock()
    viz_data = {
        "iteration": [0, 1],
        "global_best_cost": [100.0, 95.0],
        "iter_best_cost": [110.0, 105.0],
        "tau_mean": [0.5, 0.6],
        "tau_max": [0.8, 0.9],
    }
    render_policy_viz(viz_data)
    mock_st.plotly_chart.assert_called()


def test_render_policy_viz_ils():
    mock_st.reset_mock()
    viz_data = {
        "restart": [0, 1],
        "best_cost": [100.0, 95.0],
        "candidate_cost": [110.0, 105.0],
        "perturb_mode": ["shuffle", "random_swap"],
        "n_improved": [1, 0],
    }
    render_policy_viz(viz_data)
    mock_st.plotly_chart.assert_called()


def test_render_policy_viz_selector():
    mock_st.reset_mock()
    viz_data = {
        "n_selected": [5, 10],
        "mean_fill": [0.4, 0.6],
        "day": [1, 2],
    }
    render_policy_viz(viz_data)
    mock_st.plotly_chart.assert_called()


def test_render_policy_viz_rls_op_name():
    mock_st.reset_mock()
    viz_data = {
        "iteration": [0, 1],
        "op_name": ["op1", "op2"],
        "reward": [1.0, 2.0],
    }
    render_policy_viz(viz_data)
    mock_st.plotly_chart.assert_called()


def test_render_policy_viz_rls_no_op_name():
    mock_st.reset_mock()
    viz_data = {
        "iteration": [0, 1],
        "reward": [1.0, 2.0],
    }
    render_policy_viz(viz_data)
    mock_st.plotly_chart.assert_called()
