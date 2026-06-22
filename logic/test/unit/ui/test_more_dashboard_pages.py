import sys
from unittest.mock import MagicMock, patch
import pytest

# --- Setup Mock Streamlit ---
if "streamlit" in sys.modules and isinstance(sys.modules["streamlit"], MagicMock):
    mock_st = sys.modules["streamlit"]
else:
    mock_st = MagicMock()
    mock_st.session_state = {}
    mock_st.sidebar = MagicMock()
    sys.modules["streamlit"] = mock_st
    sys.modules["streamlit.components"] = MagicMock()
    sys.modules["streamlit.components.v1"] = MagicMock()

# Mock widget return values to avoid TypeErrors during comparisons
mock_st.sidebar.slider.return_value = 100
mock_st.sidebar.multiselect.return_value = []
mock_st.sidebar.selectbox.return_value = "table"
mock_st.sidebar.text_input.return_value = ""
mock_st.sidebar.radio.return_value = "Union (OR)"
mock_st.slider.return_value = 100
mock_st.selectbox.return_value = "table"
mock_st.multiselect.return_value = []
mock_st.checkbox.return_value = False
mock_st.number_input.return_value = 0
mock_st.text_input.return_value = ""

class MockContextManager:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

mock_st.spinner.return_value = MockContextManager()
mock_st.expander.return_value = MockContextManager()

def mock_columns(spec):
    count = spec if isinstance(spec, int) else len(spec)
    cols = []
    for _ in range(count):
        col = MagicMock()
        col.__enter__ = lambda self: self
        col.__exit__ = lambda self, et, ev, tb: None
        cols.append(col)
    return cols

mock_st.columns = mock_columns
mock_st.tabs = mock_columns

import pandas as pd
import numpy as np

# Imports for targets
from logic.src.ui.pages.algorithms import (
    get_tag_display_name,
    get_tag_color,
    render_algo_card,
    render_algorithms,
)
from logic.src.ui.pages.training import (
    _render_run_overview,
    _render_training_progress,
    _render_convergence_status,
    render_training_monitor,
)
from logic.src.ui.pages.training_charts import (
    _render_lr_schedule,
    _render_training_kpis,
    _render_epoch_timing,
    _render_run_comparison,
    _render_all_metrics_table,
)

from logic.src.enums.policy_tags import PolicyTag
from logic.src.enums.model_tags import ModelTag
from logic.src.enums.environment_tags import EnvironmentTag
from logic.src.enums.operator_tags import OperatorTag
from logic.src.enums.trainer_tags import TrainerTag

# =====================================================================
# Tests for algorithms.py
# =====================================================================

def test_algorithms_tags_helpers():
    assert "Policy" in get_tag_display_name(PolicyTag.CONSTRUCTION)
    assert get_tag_color(PolicyTag.CONSTRUCTION) == "#28B463"
    assert get_tag_color(ModelTag.TRANSFORMER) == "#AF7AC5"
    assert get_tag_color(EnvironmentTag.EUCLIDEAN) == "#F39C12"
    assert get_tag_color(OperatorTag.HEURISTIC) == "#E74C3C"
    assert get_tag_color(TrainerTag.REINFORCEMENT_LEARNING) == "#5D6D7E"
    assert get_tag_color("other") == "#2E86C1"

def test_render_algo_card():
    class DummyAlgo:
        """My dummy algorithm class"""
        pass

    item = {
        "name": "Dummy",
        "doc": "My dummy algorithm class",
        "tags": {PolicyTag.CONSTRUCTION, ModelTag.TRANSFORMER},
        "obj": DummyAlgo
    }
    # Should run without error
    render_algo_card(item)

@patch("logic.src.ui.pages.algorithms.GlobalRegistry")
def test_render_algorithms(mock_registry):
    class DummyAlgo:
        """Dummy docs"""
        pass
    mock_registry.get_all.return_value = {
        DummyAlgo: {PolicyTag.CONSTRUCTION}
    }
    mock_registry.get_name.return_value = "Dummy"

    # 1. OR mode
    mock_st.sidebar.multiselect.side_effect = [["Policy: CONSTRUCTION"], []]
    mock_st.sidebar.radio.return_value = "Union (OR)"
    mock_st.sidebar.text_input.return_value = ""
    render_algorithms()

    # 2. AND mode
    mock_st.sidebar.multiselect.side_effect = [["Policy: CONSTRUCTION"], []]
    mock_st.sidebar.radio.return_value = "Intersection (AND)"
    mock_st.sidebar.text_input.return_value = "Dummy"
    render_algorithms()

    # 3. Empty results / click clear
    mock_registry.get_all.return_value = {}
    mock_st.sidebar.multiselect.side_effect = None
    mock_st.sidebar.multiselect.return_value = []
    with patch("streamlit.button", return_value=True):
        render_algorithms()

# =====================================================================
# Tests for training.py
# =====================================================================

@patch("logic.src.ui.pages.training.load_hparams")
def test_render_run_overview(mock_load):
    # Empty
    mock_load.return_value = None
    _render_run_overview(["run1"])

    # Non-empty
    mock_load.return_value = {
        "env_name": "vrpp",
        "optimizer": "adam",
        "optimizer_kwargs": {"lr": 0.001},
        "baseline": "rollout",
    }
    _render_run_overview(["run1"])

@patch("logic.src.ui.pages.training.load_hparams")
def test_render_training_progress(mock_load):
    # Case 1: no hparams or no total epochs
    mock_load.return_value = {}
    _render_training_progress({}, ["run1"])

    # Case 2: valid progress
    mock_load.return_value = {"env.graph.n_days": 100}
    runs_data = {"run1": pd.DataFrame({"epoch": [0, 10, 99]})}
    _render_training_progress(runs_data, ["run1"])

def test_render_convergence_status():
    # Empty
    _render_convergence_status({"run1": pd.DataFrame()})

    # Non-empty plateau
    df = pd.DataFrame({"train/rl_loss": [0.5] * 12})
    _render_convergence_status({"run1": df})

    # Improving
    df_improving = pd.DataFrame({"train/rl_loss": list(range(20, 0, -1))})
    _render_convergence_status({"run1": df_improving})

@patch("logic.src.ui.pages.training.discover_training_runs")
@patch("logic.src.ui.pages.training.load_multiple_training_runs")
@patch("logic.src.ui.pages.training.render_training_controls")
@patch("logic.src.ui.pages.training._render_run_overview")
def test_render_training_monitor(mock_overview, mock_controls, mock_load, mock_discover):
    # No runs
    mock_discover.return_value = []
    render_training_monitor()

    # With runs
    mock_discover.return_value = [("run1", "path1")]
    mock_load.return_value = {"run1": pd.DataFrame({"epoch": [1], "step": [1], "train_loss": [0.5]})}
    mock_controls.return_value = {
        "selected_runs": ["run1"],
        "primary_metric": "train_loss",
        "secondary_metric": None,
        "x_axis": "epoch",
        "smoothing": 0.0,
    }
    with patch("logic.src.ui.pages.training.create_training_loss_chart") as mock_chart:
        render_training_monitor()

# =====================================================================
# Tests for training_charts.py
# =====================================================================

def test_render_lr_schedule():
    # No LR cols
    _render_lr_schedule({"run1": pd.DataFrame({"epoch": [1]})})

    # Yes LR cols
    df = pd.DataFrame({"step": [1, 2], "lr-Adam": [0.001, 0.0005]})
    _render_lr_schedule({"run1": df})

def test_render_training_kpis():
    runs_data = {
        "run1": pd.DataFrame({
            "epoch": [1, 2],
            "step": [10, 20],
            "train/rl_loss": [0.5, 0.4],
            "val/cost": [10.0, 9.5],
            "time/epoch_s": [5.0, 4.8],
        })
    }
    _render_training_kpis(runs_data)

def test_render_epoch_timing():
    runs_data = {
        "run1": pd.DataFrame({
            "step": [1, 2],
            "time/epoch_s": [5.0, 4.8],
        })
    }
    _render_epoch_timing(runs_data)

@patch("logic.src.ui.pages.training_charts.load_hparams")
def test_render_run_comparison(mock_load):
    mock_load.return_value = {
        "optimizer": "adam",
        "optimizer_kwargs": {"lr": 0.001},
        "batch_size": 32,
        "baseline": "rollout",
    }
    runs_data = {
        "run1": pd.DataFrame({"epoch": [1], "val_cost": [10.0]}),
        "run2": pd.DataFrame({"epoch": [1], "val_cost": [9.0]}),
    }
    _render_run_comparison(["run1", "run2"], runs_data)

def test_render_all_metrics_table():
    runs_data = {
        "run1": pd.DataFrame({"epoch": [1, 2], "loss": [0.5, 0.4]})
    }
    with patch("streamlit.multiselect", return_value=["epoch", "loss"]), \
         patch("streamlit.number_input", return_value=10):
        _render_all_metrics_table(runs_data)
