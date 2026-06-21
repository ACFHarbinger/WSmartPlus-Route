import sys
import os
import io
import pytest
from unittest.mock import MagicMock, patch

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

# Mock context managers for streamlit
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
mock_st.segmented_control = MagicMock(return_value="Raw Data")

import numpy as np
import pandas as pd
import torch

# Import targets
from logic.src.ui.pages.data_explorer import (
    _td_tensor_to_df,
    _collect_td_metadata,
    _load_td_file,
    _process_raw_to_dfs,
    _try_vrpp_split,
    _pivot_json_data,
    _load_json_file,
    _load_jsonl_file,
    _load_npz_file,
    _load_uploaded_file,
    _load_td_from_path,
    _lazy_load_td_tensor,
    _safe_nunique,
    _render_raw_data_tab,
    _render_statistics_tab,
    _render_correlation_tab,
    _render_sidebar_controls,
    _resolve_selected_df,
    render_data_explorer,
)

from logic.src.ui.pages.data_explorer_charts import (
    _resolve_column,
    _numeric_columns,
    _has_distribution_meta,
    _unique_distributions,
    _render_visualization_tab,
    _render_line_bar_chart,
    _render_scatter_chart,
    _render_area_chart,
    _render_selected_chart,
    _find_td_table,
    _render_td_coord_section,
    _render_td_dist_section,
    _render_td_overview_tab,
)

from logic.src.ui.pages.experiment_tracker import (
    _render_run_table,
    _render_run_detail,
    _render_params_table,
    _render_metric_explorer,
    _render_run_comparison,
    _render_artifacts,
    _render_dataset_events,
    _fmt_size,
    _render_tracker_sidebar,
    render_experiment_tracker,
)


# =====================================================================
# Tests for data_explorer.py
# =====================================================================

def test_td_tensor_to_df():
    # 1D
    df = _td_tensor_to_df("test", np.array([1, 2, 3]))
    assert df is not None
    assert list(df.columns) == ["test"]

    # 2D, 2 columns
    df = _td_tensor_to_df("test", np.array([[1, 2], [3, 4]]))
    assert df is not None
    assert list(df.columns) == ["x", "y"]

    # 2D, multiple columns (under _MAX_WIDE_COLS)
    df = _td_tensor_to_df("test", np.random.randn(5, 10))
    assert df is not None
    assert len(df.columns) == 10
    assert df.columns[0] == "node_0"

    # 2D, huge columns (above _MAX_WIDE_COLS)
    df = _td_tensor_to_df("test", np.random.randn(2, 250))
    assert df is not None
    assert list(df.columns) == ["sample_id", "node_id", "test"]

    # 3D, last dim = 2
    df = _td_tensor_to_df("test", np.random.randn(2, 3, 2))
    assert df is not None
    assert list(df.columns) == ["x", "y", "sample_id", "node_id"]

    # 3D, other shapes
    df = _td_tensor_to_df("test", np.random.randn(2, 3, 4))
    assert df is not None
    assert df.shape == (2, 12)


def test_collect_td_metadata():
    td = {
        "coords": torch.randn(2, 3, 2),
        "depot": torch.randn(2, 2),
        "scalar": torch.randn(2, 1),
    }
    coord_keys, depot_keys, scalar_keys, summary_rows = _collect_td_metadata(td, ["coords", "depot", "scalar"])
    assert "coords" in coord_keys
    assert "depot" in depot_keys
    assert "scalar" in scalar_keys
    assert len(summary_rows) == 3


def test_load_td_file():
    mock_st.session_state.clear()
    uploaded_file = MagicMock()
    uploaded_file.name = "test.td"
    uploaded_file.getvalue.return_value = b""

    td_mock = MagicMock()
    td_mock.batch_size = [10]
    td_mock.keys.return_value = ["coords"]
    td_mock.__getitem__.return_value = torch.randn(10, 5, 2)

    with patch("torch.load", return_value=td_mock):
        tables = _load_td_file(uploaded_file, "cache_key")
        assert len(tables) > 0
        assert mock_st.session_state["cache_key_is_td"] is True


def test_process_raw_to_dfs():
    # pd.DataFrame
    df_in = pd.DataFrame({"a": [1]})
    dfs = _process_raw_to_dfs(df_in)
    assert dfs[0] is df_in

    # Nested sequence
    dfs = _process_raw_to_dfs([[1, 2], [3, 4]])
    assert len(dfs) == 1
    assert dfs[0].shape == (2, 2)

    # NumPy array 2D
    dfs = _process_raw_to_dfs(np.array([[1, 2], [3, 4]]))
    assert len(dfs) == 1
    assert dfs[0].shape == (2, 2)


def test_try_vrpp_split():
    df = pd.DataFrame({
        0: [[1, 2]],
        1: [[3, 4]],
        2: [[[5, 6]]],
        3: [10.0]
    })
    result = _try_vrpp_split(df)
    assert result is not None
    keys = [item[0] for item in result]
    assert "Depots" in keys
    assert "Locations" in keys
    assert "Fill Values (Day 1)" in keys
    assert "Max Waste" in keys


def test_pivot_json_data():
    data = {
        "policy1_gamma1": {"cost": 10.0, "time": 5.0},
        "policy2_emp": {"cost": 12.0, "time": 4.0},
    }
    pivoted = _pivot_json_data(data, "file_id_1")
    assert pivoted["__Policy_Names__"] == ["policy1", "policy2"]
    assert pivoted["__Distributions__"] == ["gamma1", "emp"]
    assert pivoted["__File_IDs__"] == ["file_id_1", "file_id_1"]
    assert pivoted["cost"] == [10.0, 12.0]


def test_load_json_file():
    uploaded_file = MagicMock()
    uploaded_file.name = "test.json"

    # Dict form
    uploaded_file.getvalue.return_value = b'{"policy_emp": {"cost": 5.0}}'
    tables = _load_json_file(uploaded_file)
    assert len(tables) == 1

    # List form
    uploaded_file.getvalue.return_value = b'[{"cost": 5.0}, {"cost": 6.0}]'
    tables = _load_json_file(uploaded_file)
    assert len(tables) == 1


def test_load_jsonl_file():
    uploaded_file = MagicMock()
    uploaded_file.name = "test.jsonl"
    uploaded_file.getvalue.return_value = b'{"cost": 5.0}\n{"cost": 6.0}'
    tables = _load_jsonl_file(uploaded_file)
    assert len(tables) == 1
    df = list(tables.values())[0]
    assert df.shape[0] == 2


def test_load_npz_file():
    uploaded_file = MagicMock()
    # Mock np.load to return a dict-like npz
    npz_mock = MagicMock()
    npz_mock.files = ["arr1", "arr2"]
    npz_mock.__getitem__.side_effect = lambda key: np.array([1, 2, 3]) if key == "arr1" else np.array([[4, 5]])
    with patch("numpy.load", return_value=npz_mock):
        tables = _load_npz_file(uploaded_file)
        assert len(tables) == 2


def test_load_uploaded_file():
    uploaded_file = MagicMock()
    uploaded_file.name = "test.csv"
    with patch("pandas.read_csv", return_value=pd.DataFrame({"a": [1]})):
        tables = _load_uploaded_file(uploaded_file)
        assert len(tables) == 1


def test_load_td_from_path(tmp_path):
    mock_st.session_state.clear()
    td_path = tmp_path / "test.td"
    td_mock = MagicMock()
    td_mock.batch_size = [10]
    td_mock.keys.return_value = ["coords"]
    td_mock.__getitem__.return_value = torch.randn(10, 5, 2)

    with patch("torch.load", return_value=td_mock), patch("os.path.isfile", return_value=True):
        tables = _load_td_from_path(str(td_path), "cache_key")
        assert len(tables) == 1
        assert mock_st.session_state["cache_key_is_td"] is True


def test_lazy_load_td_tensor():
    mock_st.session_state.clear()
    td_meta = {"filepath": "dummy.td"}
    td_mock = MagicMock()
    td_mock.__getitem__.return_value = torch.tensor([1.0, 2.0])

    with patch("torch.load", return_value=td_mock):
        df = _lazy_load_td_tensor(td_meta, "my_key", "cache_key")
        assert df is not None
        assert mock_st.session_state["cache_key_tensor_my_key"] is df


def test_safe_nunique():
    assert _safe_nunique(pd.Series([1, 1, 2])) == 2
    assert _safe_nunique(pd.Series([{"a": 1}, {"a": 1}, {"b": 2}])) == 2


def test_render_raw_data_tab():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    _render_raw_data_tab(df, "table_name", visible_columns=["a"], row_limit=2, precision=2)


def test_render_statistics_tab():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    _render_statistics_tab(df)


def test_render_correlation_tab():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    # 2 numeric cols
    _render_correlation_tab(df)

    # 1 numeric col
    _render_correlation_tab(pd.DataFrame({"a": [1, 2, 3]}))


def test_render_sidebar_controls():
    df = pd.DataFrame({"a": [1, 2, 3]})
    opts = _render_sidebar_controls(df)
    assert "visible_columns" in opts


def test_resolve_selected_df():
    tables = {"tbl": pd.DataFrame({"a": [1]})}
    td_meta = {"filepath": "dummy.td"}

    # Exists in tables
    df = _resolve_selected_df(tables, td_meta, "tbl", "cache_key")
    assert df is not None

    # Lazy load
    td_mock = MagicMock()
    td_mock.__getitem__.return_value = torch.tensor([1.0, 2.0])
    with patch("torch.load", return_value=td_mock):
        df = _resolve_selected_df(tables, td_meta, "lazy_key", "cache_key")
        assert df is not None


def test_render_data_explorer():
    mock_st.session_state.clear()

    # 1. No input
    with patch("streamlit.radio", return_value="Upload file"), \
         patch("streamlit.file_uploader", return_value=None):
        render_data_explorer()

    # 2. Upload file path
    uploaded_file = MagicMock()
    uploaded_file.name = "test.csv"
    uploaded_file.size = 100

    with patch("streamlit.radio", return_value="Upload file"), \
         patch("streamlit.file_uploader", return_value=uploaded_file), \
         patch("pandas.read_csv", return_value=pd.DataFrame({"a": [1, 2]})), \
         patch("streamlit.selectbox", return_value="test.csv (2x1)"), \
         patch("streamlit.segmented_control", return_value="Raw Data"):
        render_data_explorer()


# =====================================================================
# Tests for data_explorer_charts.py
# =====================================================================

def test_resolve_column():
    columns = ["a", 1, "b"]
    assert _resolve_column(columns, "a") == "a"
    assert _resolve_column(columns, "1") == 1
    assert _resolve_column(columns, "c") is None


def test_numeric_columns_helper():
    df = pd.DataFrame({"a": [1], "b": ["x"]})
    assert _numeric_columns(df) == ["a"]


def test_has_distribution_meta():
    assert _has_distribution_meta(pd.DataFrame({"__Distributions__": [1]})) is True
    assert _has_distribution_meta(pd.DataFrame({"a": [1]})) is False


def test_unique_distributions():
    df = pd.DataFrame({"__Distributions__": ["gamma", "gamma", None, "emp"]})
    assert _unique_distributions(df) == ["emp", "gamma"]


def test_render_visualization_tab():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    with patch("streamlit.selectbox", side_effect=["Line Chart", "a", "b"]), \
         patch("streamlit.multiselect", return_value=["b"]):
        _render_visualization_tab(df)


def test_render_line_bar_chart():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4], "c": [5, 6]})
    # Single Y
    _render_line_bar_chart(df, "Line Chart", "a", ["b"])
    # Multi Y Line
    _render_line_bar_chart(df, "Line Chart", "a", ["b", "c"])
    # Multi Y Bar
    _render_line_bar_chart(df, "Bar Chart", "a", ["b", "c"])
    # Single Y Bar
    _render_line_bar_chart(df, "Bar Chart", "a", ["b"])


def test_render_scatter_chart():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _render_scatter_chart(df, "a", "b", {"color_by": "None", "pareto": True})


def test_render_area_chart():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _render_area_chart(df, "a", "b")


def test_render_selected_chart():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _render_selected_chart(df, "Heatmap", "a", {}, {})
    _render_selected_chart(df, "Correlation Matrix", "a", {}, {})
    _render_selected_chart(df, "Histogram", "a", {"nbins": 10}, {"y_col": "b"})
    _render_selected_chart(df, "Box Plot", "a", {}, {"y_cols": ["b"]})


def test_find_td_table():
    tables = {"my_key (10x2)": pd.DataFrame()}
    assert _find_td_table(tables, "my_key") is not None
    assert _find_td_table(tables, "other") is None


def test_render_td_coord_section():
    tables = {"coords (2x3x2)": pd.DataFrame({"x": [1, 2], "y": [3, 4], "sample_id": [0, 1]})}
    # Single Sample
    with patch("streamlit.selectbox", return_value="Single Sample"), \
         patch("streamlit.number_input", return_value=0):
        _render_td_coord_section(tables, ["coords"], ["depot"])

    # All Samples
    with patch("streamlit.selectbox", return_value="All Samples (up to 30)"):
        _render_td_coord_section(tables, ["coords"], ["depot"])


def test_render_td_dist_section():
    tables = {"scalar (3x2)": pd.DataFrame({"0": [1, 2], "1": [3, 4]})}
    _render_td_dist_section(tables, ["scalar"])


def test_render_td_overview_tab():
    td_meta = {
        "filename": "test.td",
        "batch_size": 10,
        "coord_keys": ["coords"],
        "depot_keys": ["depot"],
        "scalar_keys": ["scalar"],
        "summary_rows": [{"Key": "coords", "Shape": "(10, 5, 2)", "Dtype": "float32", "Min": 0.0, "Max": 1.0, "Mean": 0.5, "Std": 0.1}],
    }
    tables = {
        "coords (10x5x2)": pd.DataFrame({"x": [1, 2], "y": [3, 4], "sample_id": [0, 1]}),
        "scalar (10x1)": pd.DataFrame({"0": [1, 2]}),
    }
    _render_td_overview_tab(td_meta, tables)


# =====================================================================
# Tests for experiment_tracker.py
# =====================================================================

def test_fmt_size():
    assert _fmt_size(None) == "—"
    assert _fmt_size("bad") == "—"
    assert _fmt_size(500) == "500 B"
    assert _fmt_size(1500) == "1.5 KB"
    assert _fmt_size(1500000) == "1.4 MB"


def test_render_run_table():
    runs = [
        {"id": "uuid12345678", "experiment_name": "exp1", "run_type": "train", "status": "COMPLETED", "start_time": "12:00", "end_time": "12:05"},
    ]
    # No match filter
    val = _render_run_table(runs, "sim")
    assert val is None

    # Match filter
    mock_st.selectbox.return_value = "uuid1234  exp1"
    val = _render_run_table(runs, "train")
    assert val == "uuid12345678"


@patch("logic.src.ui.pages.experiment_tracker.load_run_tags")
@patch("logic.src.ui.pages.experiment_tracker.load_run_params")
def test_render_run_detail(mock_params, mock_tags):
    mock_tags.return_value = {"tag1": "val1"}
    mock_params.return_value = {
        "lr": 0.001,
        "policy_params/policy1/s0/epochs": 10
    }
    _render_run_detail("uuid1234")


def test_render_params_table():
    _render_params_table({"lr": 0.01})
    _render_params_table({})


@patch("logic.src.ui.pages.experiment_tracker.list_metric_keys")
@patch("logic.src.ui.pages.experiment_tracker.load_run_metrics")
def test_render_metric_explorer(mock_metrics, mock_keys):
    mock_keys.return_value = ["loss"]
    mock_metrics.return_value = pd.DataFrame({"step": [1, 2], "value": [0.5, 0.4]})

    with patch("streamlit.multiselect", return_value=["loss"]):
        _render_metric_explorer("uuid1234")


@patch("logic.src.ui.pages.experiment_tracker.list_metric_keys")
@patch("logic.src.ui.pages.experiment_tracker.load_run_metrics")
def test_render_run_comparison(mock_metrics, mock_keys):
    runs = [{"id": "r1", "name": "run1"}, {"id": "r2", "name": "run2"}]
    mock_keys.return_value = ["loss"]
    mock_metrics.return_value = pd.DataFrame({"step": [1, 2], "value": [0.5, 0.4]})

    with patch("streamlit.multiselect", return_value=["r1  run1", "r2  run2"]), \
         patch("streamlit.selectbox", return_value="loss"):
        _render_run_comparison(runs)


@patch("logic.src.ui.pages.experiment_tracker.load_run_artifacts")
def test_render_artifacts(mock_artifacts):
    mock_artifacts.return_value = [{"path": "model.pt", "artifact_type": "weights", "timestamp": "12:00"}]
    _render_artifacts("uuid1234")


@patch("logic.src.ui.pages.experiment_tracker.load_run_dataset_events")
def test_render_dataset_events(mock_events):
    mock_events.return_value = [
        {
            "event_type": "load",
            "file_path": "data.pkl",
            "shape": "(100, 2)",
            "size_bytes": 2048,
            "timestamp": "12:00",
            "metadata": '{"source_file": "loader.py", "source_line": 42, "variable_name": "data"}'
        }
    ]
    _render_dataset_events("uuid1234")


def test_render_tracker_sidebar():
    res = _render_tracker_sidebar(["train", "sim"])
    assert res["run_type_filter"] is not None


@patch("logic.src.ui.pages.experiment_tracker.load_tracking_runs")
@patch("logic.src.ui.pages.experiment_tracker._render_run_table")
@patch("logic.src.ui.pages.experiment_tracker._render_run_detail")
@patch("logic.src.ui.pages.experiment_tracker._render_metric_explorer")
@patch("logic.src.ui.pages.experiment_tracker._render_artifacts")
@patch("logic.src.ui.pages.experiment_tracker._render_dataset_events")
@patch("logic.src.ui.pages.experiment_tracker._render_run_comparison")
@patch("logic.src.ui.pages.experiment_tracker._render_mlflow_explorer")
@patch("logic.src.ui.pages.experiment_tracker._render_zenml_pipelines")
def test_render_experiment_tracker(
    mock_zenml, mock_mlflow, mock_compare, mock_events, mock_artifacts, mock_metric, mock_detail, mock_table, mock_runs
):
    mock_runs.return_value = [{"id": "uuid1234", "run_type": "train"}]
    mock_table.return_value = "uuid1234"

    render_experiment_tracker()


# =====================================================================
# Tests for experiment_tracker_mlflow.py & experiment_tracker_zenml.py
# =====================================================================

from logic.src.ui.pages.experiment_tracker_mlflow import _render_mlflow_explorer
from logic.src.ui.pages.experiment_tracker_zenml import _render_zenml_pipelines

@patch("logic.src.ui.pages.experiment_tracker_mlflow.load_mlflow_runs")
@patch("logic.src.ui.pages.experiment_tracker_mlflow.list_mlflow_metric_keys")
@patch("logic.src.ui.pages.experiment_tracker_mlflow.load_mlflow_metric_history")
def test_render_mlflow_explorer(mock_history, mock_keys, mock_runs):
    # Case 1: No runs
    mock_runs.return_value = None
    _render_mlflow_explorer("http://uri", "exp")

    # Case 2: Yes runs
    df_runs = pd.DataFrame({
        "run_id": ["r1"],
        "params.lr": ["0.01"],
        "metric.loss": [0.5],
    })
    mock_runs.return_value = df_runs
    mock_keys.return_value = ["loss"]
    mock_history.return_value = pd.DataFrame({"step": [1, 2], "value": [0.5, 0.4]})

    with patch("streamlit.selectbox", return_value="r1"), \
         patch("streamlit.multiselect", return_value=["loss"]), \
         patch("streamlit.expander", return_value=MockContextManager()):
        _render_mlflow_explorer("http://uri", "exp")


@patch("logic.src.ui.pages.experiment_tracker_zenml.load_zenml_pipeline_runs")
@patch("logic.src.ui.pages.experiment_tracker_zenml.load_zenml_run_steps")
def test_render_zenml_pipelines(mock_steps, mock_runs):
    # Case 1: No runs
    mock_runs.return_value = None
    _render_zenml_pipelines()

    # Case 2: Yes runs
    mock_runs.return_value = [
        {"id": "zen12345678", "pipeline": "my_pipe", "status": "completed"}
    ]
    mock_steps.return_value = [
        {"step_name": "load_data", "status": "completed"},
        {"step_name": "train", "status": "failed"},
    ]

    with patch("streamlit.selectbox", return_value="zen12345  my_pipe"):
        _render_zenml_pipelines()
