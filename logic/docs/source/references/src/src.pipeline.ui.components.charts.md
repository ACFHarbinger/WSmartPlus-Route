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
* - {py:obj}`create_training_loss_chart <src.pipeline.ui.components.charts.create_training_loss_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_training_loss_chart
    :summary:
    ```
* - {py:obj}`create_simulation_metrics_chart <src.pipeline.ui.components.charts.create_simulation_metrics_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_simulation_metrics_chart
    :summary:
    ```
* - {py:obj}`create_kpi_cards_html <src.pipeline.ui.components.charts.create_kpi_cards_html>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_kpi_cards_html
    :summary:
    ```
* - {py:obj}`create_bar_chart <src.pipeline.ui.components.charts.create_bar_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.charts.create_bar_chart
    :summary:
    ```
````

### API

````{py:function} apply_moving_average(data: pandas.Series, window: int) -> pandas.Series
:canonical: src.pipeline.ui.components.charts.apply_moving_average

```{autodoc2-docstring} src.pipeline.ui.components.charts.apply_moving_average
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

````{py:function} create_kpi_cards_html(metrics: typing.Dict[str, typing.Any], prefix: str = '') -> str
:canonical: src.pipeline.ui.components.charts.create_kpi_cards_html

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_kpi_cards_html
```
````

````{py:function} create_bar_chart(data: typing.Dict[str, float], title: str = 'Comparison', x_label: str = 'Category', y_label: str = 'Value') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.charts.create_bar_chart

```{autodoc2-docstring} src.pipeline.ui.components.charts.create_bar_chart
```
````
