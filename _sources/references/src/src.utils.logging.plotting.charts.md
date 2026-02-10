# {py:mod}`src.utils.logging.plotting.charts`

```{py:module} src.utils.logging.plotting.charts
```

```{autodoc2-docstring} src.utils.logging.plotting.charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`plot_linechart <src.utils.logging.plotting.charts.plot_linechart>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts.plot_linechart
    :summary:
    ```
* - {py:obj}`_plot_2d_graph <src.utils.logging.plotting.charts._plot_2d_graph>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._plot_2d_graph
    :summary:
    ```
* - {py:obj}`_annotate_plot <src.utils.logging.plotting.charts._annotate_plot>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._annotate_plot
    :summary:
    ```
* - {py:obj}`_set_plot_attributes <src.utils.logging.plotting.charts._set_plot_attributes>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._set_plot_attributes
    :summary:
    ```
* - {py:obj}`_save_plot <src.utils.logging.plotting.charts._save_plot>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._save_plot
    :summary:
    ```
* - {py:obj}`_add_scatter_marker <src.utils.logging.plotting.charts._add_scatter_marker>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._add_scatter_marker
    :summary:
    ```
* - {py:obj}`_plot_pareto_front <src.utils.logging.plotting.charts._plot_pareto_front>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._plot_pareto_front
    :summary:
    ```
* - {py:obj}`_calculate_dominance <src.utils.logging.plotting.charts._calculate_dominance>`
  - ```{autodoc2-docstring} src.utils.logging.plotting.charts._calculate_dominance
    :summary:
    ```
````

### API

````{py:function} plot_linechart(output_dest, graph_log, plot_func, policies, x_label=None, y_label=None, title=None, fsave=True, scale='linear', x_values=None, linestyles=None, markers=None, annotate=True, pareto_front=False)
:canonical: src.utils.logging.plotting.charts.plot_linechart

```{autodoc2-docstring} src.utils.logging.plotting.charts.plot_linechart
```
````

````{py:function} _plot_2d_graph(plot_func, graph_log, markers) -> typing.Dict[int, typing.List[typing.Tuple[float, float]]]
:canonical: src.utils.logging.plotting.charts._plot_2d_graph

```{autodoc2-docstring} src.utils.logging.plotting.charts._plot_2d_graph
```
````

````{py:function} _annotate_plot(graph_log) -> None
:canonical: src.utils.logging.plotting.charts._annotate_plot

```{autodoc2-docstring} src.utils.logging.plotting.charts._annotate_plot
```
````

````{py:function} _set_plot_attributes(scale, x_label, y_label, policies) -> None
:canonical: src.utils.logging.plotting.charts._set_plot_attributes

```{autodoc2-docstring} src.utils.logging.plotting.charts._set_plot_attributes
```
````

````{py:function} _save_plot(output_dest, x_values) -> None
:canonical: src.utils.logging.plotting.charts._save_plot

```{autodoc2-docstring} src.utils.logging.plotting.charts._save_plot
```
````

````{py:function} _add_scatter_marker(xy: typing.Tuple[float, float]) -> None
:canonical: src.utils.logging.plotting.charts._add_scatter_marker

```{autodoc2-docstring} src.utils.logging.plotting.charts._add_scatter_marker
```
````

````{py:function} _plot_pareto_front(points: typing.List[typing.Tuple[float, float]], dominance_ls: typing.List[int], id_nbins: int) -> None
:canonical: src.utils.logging.plotting.charts._plot_pareto_front

```{autodoc2-docstring} src.utils.logging.plotting.charts._plot_pareto_front
```
````

````{py:function} _calculate_dominance(points: typing.List[typing.Tuple[float, float]]) -> typing.List[int]
:canonical: src.utils.logging.plotting.charts._calculate_dominance

```{autodoc2-docstring} src.utils.logging.plotting.charts._calculate_dominance
```
````
