import sys
from unittest.mock import MagicMock

import pandas as pd
from logic.src.ui.pages.data_explorer_charts import (
    _find_td_table,
    _has_distribution_meta,
    _numeric_columns,
    _render_area_chart,
    _render_line_bar_chart,
    _render_scatter_chart,
    _render_td_coord_section,
    _render_td_dist_section,
    _render_td_overview_tab,
    _render_visualization_tab,
    _resolve_column,
    _unique_distributions,
)


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

# Ensure streamlit.components.v1 exists in sys.modules to satisfy external imports
sys.modules["streamlit.components"] = MagicMock()
sys.modules["streamlit.components.v1"] = MagicMock()


# Dynamically return appropriate number of columns
def mock_columns(num_or_spec):
    num = num_or_spec if isinstance(num_or_spec, int) else len(num_or_spec)
    return [MagicMock() for _ in range(num)]


mock_st.columns = mock_columns

# Implement side effects for input widgets
mock_st.selectbox.side_effect = lambda label, options, index=0, **kwargs: (
    options[index] if index < len(options) else (options[0] if options else "")
)
mock_st.multiselect.side_effect = lambda label, options, default=None, **kwargs: (
    default if default is not None else ([options[0]] if options else [])
)
mock_st.slider.side_effect = lambda label, min_value, max_value, value, **kwargs: value
mock_st.number_input.side_effect = lambda label, min_value=0, max_value=1, value=0, **kwargs: value
mock_st.checkbox.side_effect = lambda label, value=False, **kwargs: value


def test_resolve_column():
    columns = ["a", 2, "c"]
    assert _resolve_column(columns, "a") == "a"
    assert _resolve_column(columns, "2") == 2
    assert _resolve_column(columns, "nonexistent") is None


def test_numeric_columns():
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"], "c": [1.1, 2.2, 3.3]})
    assert _numeric_columns(df) == ["a", "c"]


def test_has_distribution_meta_and_unique_distributions():
    df1 = pd.DataFrame({"a": [1]})
    assert not _has_distribution_meta(df1)
    assert _unique_distributions(df1) == []

    df2 = pd.DataFrame({"__Distributions__": ["d1", "d2", "d1", None]})
    assert _has_distribution_meta(df2)
    assert _unique_distributions(df2) == ["d1", "d2"]


def test_render_line_bar_chart():
    df = pd.DataFrame({"x_col": [1, 2, 3], "y_col": [10.0, 20.0, 30.0], "y_col2": [100, 200, 300]})

    # Line Chart with single Y
    mock_st.reset_mock()
    _render_line_bar_chart(df, "Line Chart", "x_col", ["y_col"])
    mock_st.plotly_chart.assert_called()

    # Line Chart with multiple Y
    mock_st.reset_mock()
    _render_line_bar_chart(df, "Line Chart", "x_col", ["y_col", "y_col2"])
    mock_st.plotly_chart.assert_called()

    # Bar Chart with single Y
    mock_st.reset_mock()
    _render_line_bar_chart(df, "Bar Chart", "x_col", ["y_col"])
    mock_st.plotly_chart.assert_called()

    # Bar Chart with multiple Y
    mock_st.reset_mock()
    _render_line_bar_chart(df, "Bar Chart", "x_col", ["y_col", "y_col2"])
    mock_st.plotly_chart.assert_called()

    # Warnings / Errors
    mock_st.reset_mock()
    _render_line_bar_chart(df, "Line Chart", "x_col", [])
    mock_st.warning.assert_called()

    mock_st.reset_mock()
    _render_line_bar_chart(df, "Line Chart", "nonexistent_x", ["y_col"])
    mock_st.error.assert_called()


def test_render_scatter_chart():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0], "category": ["A", "B", "A"]})

    # Basic scatter
    mock_st.reset_mock()
    _render_scatter_chart(df, "x", "y", {"color_by": "None", "pareto": False})
    mock_st.plotly_chart.assert_called()

    # Pareto and color_by scatter
    mock_st.reset_mock()
    _render_scatter_chart(df, "x", "y", {"color_by": "category", "pareto": True})
    mock_st.plotly_chart.assert_called()

    # Error
    mock_st.reset_mock()
    _render_scatter_chart(df, "nonexistent", "y", {})
    mock_st.error.assert_called()


def test_render_area_chart():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30]})
    mock_st.reset_mock()
    _render_area_chart(df, "x", "y")
    mock_st.plotly_chart.assert_called()

    # Error
    mock_st.reset_mock()
    _render_area_chart(df, "nonexistent", "y")
    mock_st.error.assert_called()


def test_render_visualization_tab():
    df = pd.DataFrame({"x": [1, 2, 3], "y": [10, 20, 30], "__Distributions__": ["d1", "d1", "d1"]})
    for chart_type in [
        "Line Chart",
        "Bar Chart",
        "Scatter Plot",
        "Area Chart",
        "Histogram",
        "Box Plot",
        "Heatmap",
        "Correlation Matrix",
    ]:
        mock_st.reset_mock()

        # Mock st.selectbox to return the specific chart type
        def selectbox_side_effect(label, options, index=0, _ct=chart_type, **kwargs):
            if label == "Chart Type":
                return _ct
            elif label == "Distribution":
                return "d1"
            elif label == "X Axis":
                return "x"
            elif label == "Y Axis" or label == "Column":
                return "y"
            elif label == "Color by":
                return "None"
            return options[index] if index < len(options) else (options[0] if options else "")

        mock_st.selectbox.side_effect = selectbox_side_effect
        _render_visualization_tab(df)
        mock_st.plotly_chart.assert_called()


def test_find_td_table():
    tables = {"params (10x10)": pd.DataFrame({"x": [1]}), "coords": pd.DataFrame({"y": [2]})}
    assert _find_td_table(tables, "params") is not None
    assert _find_td_table(tables, "nonexistent") is None


def test_render_td_coord_section():
    tables = {
        "coords (10x2)": pd.DataFrame({"sample_id": [0, 1], "x": [1.0, 2.0], "y": [10.0, 20.0]}),
        "depot (1x2)": pd.DataFrame({"x": [1.5], "y": [15.0]}),
    }
    coord_keys = ["coords"]
    depot_keys = ["depot"]

    # Single Sample mode
    mock_st.reset_mock()
    mock_st.selectbox.side_effect = lambda label, options, index=0, **kwargs: (
        "Single Sample" if label == "Display Mode" else "coords"
    )
    _render_td_coord_section(tables, coord_keys, depot_keys)
    mock_st.plotly_chart.assert_called()

    # All Samples mode
    mock_st.reset_mock()
    mock_st.selectbox.side_effect = lambda label, options, index=0, **kwargs: (
        "All Samples (up to 30)" if label == "Display Mode" else "coords"
    )
    _render_td_coord_section(tables, coord_keys, depot_keys)
    mock_st.plotly_chart.assert_called()


def test_render_td_dist_section():
    tables = {"scalars (10x5)": pd.DataFrame([[1, 2], [3, 4]], columns=["node1", "node2"])}
    scalar_keys = ["scalars"]

    mock_st.reset_mock()
    mock_st.selectbox.side_effect = lambda label, options, index=0, **kwargs: "scalars"
    _render_td_dist_section(tables, scalar_keys)
    # Check that plotly_chart was called twice (once for histogram, once for box plot)
    assert mock_st.plotly_chart.call_count == 2


def test_render_td_overview_tab():
    td_meta = {
        "filename": "test_instances.td",
        "batch_size": 100,
        "coord_keys": ["coords"],
        "depot_keys": ["depot"],
        "scalar_keys": ["scalars"],
        "summary_rows": [{"Key": "coords", "Shape": "[100, 50, 2]"}],
    }
    tables = {
        "coords (100x2)": pd.DataFrame({"sample_id": [0], "x": [1.0], "y": [10.0]}),
        "depot (100x2)": pd.DataFrame({"x": [1.5], "y": [15.0]}),
        "scalars (100x1)": pd.DataFrame({"val": [5.0]}),
    }

    mock_st.reset_mock()
    _render_td_overview_tab(td_meta, tables)
    mock_st.dataframe.assert_called()
    assert mock_st.plotly_chart.call_count >= 2
