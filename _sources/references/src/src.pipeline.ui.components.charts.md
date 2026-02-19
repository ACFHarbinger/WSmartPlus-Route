# {py:mod}`src.pipeline.ui.components.charts`

```{py:module} src.pipeline.ui.components.charts
```

```{autodoc2-docstring} src.pipeline.ui.components.charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_moving_average <src.pipeline.ui.components.charts.apply_moving_average>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.apply_moving_average
    :summary:
    ```
* - {py:obj}`create_sparkline_svg <src.pipeline.ui.components.charts.create_sparkline_svg>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_sparkline_svg
    :summary:
    ```
* - {py:obj}`create_training_loss_chart <src.pipeline.ui.components.charts.create_training_loss_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_training_loss_chart
    :summary:
    ```
* - {py:obj}`create_simulation_metrics_chart <src.pipeline.ui.components.charts.create_simulation_metrics_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_simulation_metrics_chart
    :summary:
    ```
* - {py:obj}`create_radar_chart <src.pipeline.ui.components.charts.create_radar_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_radar_chart
    :summary:
    ```
* - {py:obj}`create_stacked_bar_chart <src.pipeline.ui.components.charts.create_stacked_bar_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_stacked_bar_chart
    :summary:
    ```
* - {py:obj}`create_heatmap_chart <src.pipeline.ui.components.charts.create_heatmap_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_heatmap_chart
    :summary:
    ```
* - {py:obj}`create_area_chart <src.pipeline.ui.components.charts.create_area_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_area_chart
    :summary:
    ```
* - {py:obj}`create_bar_chart <src.pipeline.ui.components.charts.create_bar_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_bar_chart
    :summary:
    ```
* - {py:obj}`calculate_pareto_front <src.pipeline.ui.components.charts.calculate_pareto_front>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.calculate_pareto_front
    :summary:
    ```
* - {py:obj}`create_histogram_chart <src.pipeline.ui.components.charts.create_histogram_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_histogram_chart
    :summary:
    ```
* - {py:obj}`create_box_plot_chart <src.pipeline.ui.components.charts.create_box_plot_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_box_plot_chart
    :summary:
    ```
* - {py:obj}`create_correlation_matrix_chart <src.pipeline.ui.components.charts.create_correlation_matrix_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_correlation_matrix_chart
    :summary:
    ```
* - {py:obj}`create_pareto_scatter_chart <src.pipeline.ui.components.charts.create_pareto_scatter_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_pareto_scatter_chart
    :summary:
    ```
* - {py:obj}`create_multi_y_line_chart <src.pipeline.ui.components.charts.create_multi_y_line_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_multi_y_line_chart
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PLOTLY_LAYOUT_DEFAULTS <src.pipeline.ui.components.charts.PLOTLY_LAYOUT_DEFAULTS>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.PLOTLY_LAYOUT_DEFAULTS
    :summary:
    ```
````

### API

````{py:data} PLOTLY_LAYOUT_DEFAULTS
:canonical: src.pipeline.ui.components.charts.PLOTLY_LAYOUT_DEFAULTS
:type: typing.Dict[str, typing.Any]
:value: >
   None

```{autodoc2-docstring} src.pipeline.ui.components.charts.PLOTLY_LAYOUT_DEFAULTS
```

````

````{py:function} apply_moving_average(data: pandas.Series, window: int) -> pandas.Series
:canonical: src.pipeline.ui.components.charts.apply_moving_average

```{autodoc2-docstring} src.pipeline.ui.components.charts.apply_moving_average
```
````

````{py:function} create_sparkline_svg(values: typing.List[float], width: int = 60, height: int = 20, color: str = 'rgba(255,255,255,0.8)') -> str
:canonical: src.pipeline.ui.components.charts.create_sparkline_svg

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_sparkline_svg
```
````

````{py:function} create_training_loss_chart(runs_data: typing.Dict[str, pandas.DataFrame], metric_y1: str = 'train_loss', metric_y2: typing.Optional[str] = 'val_cost', x_axis: str = 'epoch', smoothing: int = 1) -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_training_loss_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_training_loss_chart
```
````

````{py:function} create_simulation_metrics_chart(df: pandas.DataFrame, metrics: typing.List[str], x_axis: str = 'day', show_std: bool = True) -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_simulation_metrics_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_simulation_metrics_chart
```
````

````{py:function} create_radar_chart(policy_metrics: typing.Dict[str, typing.Dict[str, float]], metrics: typing.List[str]) -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_radar_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_radar_chart
```
````

````{py:function} create_stacked_bar_chart(categories: typing.List[str], series: typing.Dict[str, typing.List[float]], title: str = '', x_label: str = '', y_label: str = '', colors: typing.Optional[typing.List[str]] = None) -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_stacked_bar_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_stacked_bar_chart
```
````

````{py:function} create_heatmap_chart(df: pandas.DataFrame, title: str = 'Heatmap') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_heatmap_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_heatmap_chart
```
````

````{py:function} create_area_chart(x: typing.Any, y: typing.Any, x_label: str = 'X', y_label: str = 'Y', title: str = 'Area Chart') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_area_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_area_chart
```
````

````{py:function} create_bar_chart(data: typing.Dict[str, float], title: str = 'Comparison', x_label: str = 'Category', y_label: str = 'Value') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_bar_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_bar_chart
```
````

````{py:function} calculate_pareto_front(x_values: typing.List[float], y_values: typing.List[float]) -> typing.List[int]
:canonical: src.pipeline.ui.components.charts.calculate_pareto_front

```{autodoc2-docstring} src.pipeline.ui.components.charts.calculate_pareto_front
```
````

````{py:function} create_histogram_chart(series: pandas.Series, nbins: int = 30, title: str = 'Histogram') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_histogram_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_histogram_chart
```
````

````{py:function} create_box_plot_chart(df: pandas.DataFrame, columns: typing.List[str], title: str = 'Box Plot') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_box_plot_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_box_plot_chart
```
````

````{py:function} create_correlation_matrix_chart(df: pandas.DataFrame, title: str = 'Correlation Matrix') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_correlation_matrix_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_correlation_matrix_chart
```
````

````{py:function} create_pareto_scatter_chart(x: typing.Any, y: typing.Any, x_label: str = 'X', y_label: str = 'Y', pareto_indices: typing.Optional[typing.List[int]] = None, color_by: typing.Optional[pandas.Series] = None, title: str = 'Scatter with Pareto Front') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_pareto_scatter_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_pareto_scatter_chart
```
````

````{py:function} create_multi_y_line_chart(df: pandas.DataFrame, x_col: str, y_cols: typing.List[str], title: str = 'Multi-Series Line Chart') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_multi_y_line_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_multi_y_line_chart
```
````
