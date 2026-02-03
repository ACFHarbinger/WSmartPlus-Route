# {py:mod}`src.pipeline.ui.components.benchmark_charts`

```{py:module} src.pipeline.ui.components.benchmark_charts
```

```{autodoc2-docstring} src.pipeline.ui.components.benchmark_charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_benchmark_comparison_chart <src.pipeline.ui.components.benchmark_charts.create_benchmark_comparison_chart>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.benchmark_charts.create_benchmark_comparison_chart
    :summary:
    ```
* - {py:obj}`create_latency_throughput_scatter <src.pipeline.ui.components.benchmark_charts.create_latency_throughput_scatter>`
  - ```{autodoc2-docstring} src.pipeline.ui.components.benchmark_charts.create_latency_throughput_scatter
    :summary:
    ```
````

### API

````{py:function} create_benchmark_comparison_chart(df: pandas.DataFrame, metric: str, title: str, x_axis: str = 'policy', color_col: str = 'num_nodes') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.benchmark_charts.create_benchmark_comparison_chart

```{autodoc2-docstring} src.pipeline.ui.components.benchmark_charts.create_benchmark_comparison_chart
```
````

````{py:function} create_latency_throughput_scatter(df: pandas.DataFrame, latency_col: str = 'latency', throughput_col: str = 'throughput', color_col: str = 'num_nodes') -> plotly.graph_objects.Figure
:canonical: src.pipeline.ui.components.benchmark_charts.create_latency_throughput_scatter

```{autodoc2-docstring} src.pipeline.ui.components.benchmark_charts.create_latency_throughput_scatter
```
````
