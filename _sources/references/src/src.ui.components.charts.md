# {py:mod}`src.ui.components.charts`

```{py:module} src.ui.components.charts
```

```{autodoc2-docstring} src.ui.components.charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`apply_moving_average <src.ui.components.charts.apply_moving_average>`
  - ```{autodoc2-docstring} src.ui.components.charts.apply_moving_average
    :summary:
    ```
* - {py:obj}`create_sparkline_svg <src.ui.components.charts.create_sparkline_svg>`
  - ```{autodoc2-docstring} src.ui.components.charts.create_sparkline_svg
    :summary:
    ```
* - {py:obj}`create_training_loss_chart <src.ui.components.charts.create_training_loss_chart>`
  - ```{autodoc2-docstring} src.ui.components.charts.create_training_loss_chart
    :summary:
    ```
* - {py:obj}`create_simulation_metrics_chart <src.ui.components.charts.create_simulation_metrics_chart>`
  - ```{autodoc2-docstring} src.ui.components.charts.create_simulation_metrics_chart
    :summary:
    ```
* - {py:obj}`create_radar_chart <src.ui.components.charts.create_radar_chart>`
  - ```{autodoc2-docstring} src.ui.components.charts.create_radar_chart
    :summary:
    ```
* - {py:obj}`create_stacked_bar_chart <src.ui.components.charts.create_stacked_bar_chart>`
  - ```{autodoc2-docstring} src.ui.components.charts.create_stacked_bar_chart
    :summary:
    ```
* - {py:obj}`create_heatmap_chart <src.ui.components.charts.create_heatmap_chart>`
  - ```{autodoc2-docstring} src.ui.components.charts.create_heatmap_chart
    :summary:
    ```
````

### API

````{py:function} apply_moving_average(data: pandas.Series, window: int) -> pandas.Series
:canonical: src.ui.components.charts.apply_moving_average

```{autodoc2-docstring} src.ui.components.charts.apply_moving_average
```
````

````{py:function} create_sparkline_svg(values: typing.List[float], width: int = 60, height: int = 20, color: str = 'rgba(255,255,255,0.8)') -> str
:canonical: src.ui.components.charts.create_sparkline_svg

```{autodoc2-docstring} src.ui.components.charts.create_sparkline_svg
```
````

````{py:function} create_training_loss_chart(runs_data: typing.Dict[str, pandas.DataFrame], metric_y1: str = 'train_loss', metric_y2: typing.Optional[str] = 'val_cost', x_axis: str = 'epoch', smoothing: int = 1) -> plotly.graph_objects.Figure
:canonical: src.ui.components.charts.create_training_loss_chart

```{autodoc2-docstring} src.ui.components.charts.create_training_loss_chart
```
````

````{py:function} create_simulation_metrics_chart(df: pandas.DataFrame, metrics: typing.List[str], x_axis: str = 'day', show_std: bool = True) -> plotly.graph_objects.Figure
:canonical: src.ui.components.charts.create_simulation_metrics_chart

```{autodoc2-docstring} src.ui.components.charts.create_simulation_metrics_chart
```
````

````{py:function} create_radar_chart(policy_metrics: typing.Dict[str, typing.Dict[str, float]], metrics: typing.List[str]) -> plotly.graph_objects.Figure
:canonical: src.ui.components.charts.create_radar_chart

```{autodoc2-docstring} src.ui.components.charts.create_radar_chart
```
````

````{py:function} create_stacked_bar_chart(categories: typing.List[str], series: typing.Dict[str, typing.List[float]], title: str = '', x_label: str = '', y_label: str = '', colors: typing.Optional[typing.List[str]] = None) -> plotly.graph_objects.Figure
:canonical: src.ui.components.charts.create_stacked_bar_chart

```{autodoc2-docstring} src.ui.components.charts.create_stacked_bar_chart
```
````

````{py:function} create_heatmap_chart(df: pandas.DataFrame, title: str = 'Heatmap') -> plotly.graph_objects.Figure
:canonical: src.ui.components.charts.create_heatmap_chart

```{autodoc2-docstring} src.ui.components.charts.create_heatmap_chart
```
````
