# {py:mod}`src.ui.components.explorer_charts`

```{py:module} src.ui.components.explorer_charts
```

```{autodoc2-docstring} src.ui.components.explorer_charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_area_chart <src.ui.components.explorer_charts.create_area_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_area_chart
    :summary:
    ```
* - {py:obj}`create_bar_chart <src.ui.components.explorer_charts.create_bar_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_bar_chart
    :summary:
    ```
* - {py:obj}`calculate_pareto_front <src.ui.components.explorer_charts.calculate_pareto_front>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.calculate_pareto_front
    :summary:
    ```
* - {py:obj}`create_histogram_chart <src.ui.components.explorer_charts.create_histogram_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_histogram_chart
    :summary:
    ```
* - {py:obj}`create_box_plot_chart <src.ui.components.explorer_charts.create_box_plot_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_box_plot_chart
    :summary:
    ```
* - {py:obj}`create_correlation_matrix_chart <src.ui.components.explorer_charts.create_correlation_matrix_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_correlation_matrix_chart
    :summary:
    ```
* - {py:obj}`create_pareto_scatter_chart <src.ui.components.explorer_charts.create_pareto_scatter_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_pareto_scatter_chart
    :summary:
    ```
* - {py:obj}`create_multi_y_line_chart <src.ui.components.explorer_charts.create_multi_y_line_chart>`
  - ```{autodoc2-docstring} src.ui.components.explorer_charts.create_multi_y_line_chart
    :summary:
    ```
````

### API

````{py:function} create_area_chart(x: typing.Any, y: typing.Any, x_label: str = 'X', y_label: str = 'Y', title: str = 'Area Chart') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_area_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_area_chart
```
````

````{py:function} create_bar_chart(data: typing.Dict[str, float], title: str = 'Comparison', x_label: str = 'Category', y_label: str = 'Value') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_bar_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_bar_chart
```
````

````{py:function} calculate_pareto_front(x_values: typing.List[float], y_values: typing.List[float]) -> typing.List[int]
:canonical: src.ui.components.explorer_charts.calculate_pareto_front

```{autodoc2-docstring} src.ui.components.explorer_charts.calculate_pareto_front
```
````

````{py:function} create_histogram_chart(series: pandas.Series, nbins: int = 30, title: str = 'Histogram') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_histogram_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_histogram_chart
```
````

````{py:function} create_box_plot_chart(df: pandas.DataFrame, columns: typing.List[str], title: str = 'Box Plot') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_box_plot_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_box_plot_chart
```
````

````{py:function} create_correlation_matrix_chart(df: pandas.DataFrame, title: str = 'Correlation Matrix') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_correlation_matrix_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_correlation_matrix_chart
```
````

````{py:function} create_pareto_scatter_chart(x: typing.Any, y: typing.Any, x_label: str = 'X', y_label: str = 'Y', pareto_indices: typing.Optional[typing.List[int]] = None, color_by: typing.Optional[pandas.Series] = None, title: str = 'Scatter with Pareto Front') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_pareto_scatter_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_pareto_scatter_chart
```
````

````{py:function} create_multi_y_line_chart(df: pandas.DataFrame, x_col: str, y_cols: typing.List[str], title: str = 'Multi-Series Line Chart') -> plotly.graph_objects.Figure
:canonical: src.ui.components.explorer_charts.create_multi_y_line_chart

```{autodoc2-docstring} src.ui.components.explorer_charts.create_multi_y_line_chart
```
````
