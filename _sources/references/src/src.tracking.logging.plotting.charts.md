# {py:mod}`src.tracking.logging.plotting.charts`

```{py:module} src.tracking.logging.plotting.charts
```

```{autodoc2-docstring} src.tracking.logging.plotting.charts
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`plot_linechart <src.tracking.logging.plotting.charts.plot_linechart>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts.plot_linechart
    :summary:
    ```
* - {py:obj}`_plot_2d_graph <src.tracking.logging.plotting.charts._plot_2d_graph>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._plot_2d_graph
    :summary:
    ```
* - {py:obj}`_annotate_plot <src.tracking.logging.plotting.charts._annotate_plot>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._annotate_plot
    :summary:
    ```
* - {py:obj}`_set_plot_attributes <src.tracking.logging.plotting.charts._set_plot_attributes>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._set_plot_attributes
    :summary:
    ```
* - {py:obj}`_save_plot <src.tracking.logging.plotting.charts._save_plot>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._save_plot
    :summary:
    ```
* - {py:obj}`_add_scatter_marker <src.tracking.logging.plotting.charts._add_scatter_marker>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._add_scatter_marker
    :summary:
    ```
* - {py:obj}`_plot_pareto_front <src.tracking.logging.plotting.charts._plot_pareto_front>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._plot_pareto_front
    :summary:
    ```
* - {py:obj}`_calculate_dominance <src.tracking.logging.plotting.charts._calculate_dominance>`
  - ```{autodoc2-docstring} src.tracking.logging.plotting.charts._calculate_dominance
    :summary:
    ```
````

### API

````{py:function} plot_linechart(output_dest: str, graph_log: numpy.ndarray, plot_func: typing.Callable[..., typing.Any], policies: typing.List[str], x_label: typing.Optional[str] = None, y_label: typing.Optional[str] = None, title: typing.Optional[str] = None, fsave: bool = True, scale: str = 'linear', x_values: typing.Optional[typing.Union[typing.List[float], numpy.ndarray]] = None, linestyles: typing.Optional[typing.List[str]] = None, markers: typing.Optional[typing.List[str]] = None, annotate: bool = True, pareto_front: bool = False) -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.tracking.logging.plotting.charts.plot_linechart

```{autodoc2-docstring} src.tracking.logging.plotting.charts.plot_linechart
```
````

````{py:function} _plot_2d_graph(plot_func: typing.Callable[..., typing.Any], graph_log: numpy.ndarray, markers: typing.Optional[typing.List[str]]) -> typing.Dict[int, typing.List[typing.Tuple[float, float]]]
:canonical: src.tracking.logging.plotting.charts._plot_2d_graph

```{autodoc2-docstring} src.tracking.logging.plotting.charts._plot_2d_graph
```
````

````{py:function} _annotate_plot(graph_log: numpy.ndarray) -> None
:canonical: src.tracking.logging.plotting.charts._annotate_plot

```{autodoc2-docstring} src.tracking.logging.plotting.charts._annotate_plot
```
````

````{py:function} _set_plot_attributes(scale: str, x_label: typing.Optional[str], y_label: typing.Optional[str], policies: typing.List[str]) -> None
:canonical: src.tracking.logging.plotting.charts._set_plot_attributes

```{autodoc2-docstring} src.tracking.logging.plotting.charts._set_plot_attributes
```
````

````{py:function} _save_plot(output_dest: str, x_values: typing.Optional[typing.Union[typing.List[float], numpy.ndarray]]) -> None
:canonical: src.tracking.logging.plotting.charts._save_plot

```{autodoc2-docstring} src.tracking.logging.plotting.charts._save_plot
```
````

````{py:function} _add_scatter_marker(xy: typing.Tuple[float, float]) -> None
:canonical: src.tracking.logging.plotting.charts._add_scatter_marker

```{autodoc2-docstring} src.tracking.logging.plotting.charts._add_scatter_marker
```
````

````{py:function} _plot_pareto_front(points: typing.List[typing.Tuple[float, float]], dominance_ls: typing.List[int], id_nbins: int) -> None
:canonical: src.tracking.logging.plotting.charts._plot_pareto_front

```{autodoc2-docstring} src.tracking.logging.plotting.charts._plot_pareto_front
```
````

````{py:function} _calculate_dominance(points: typing.List[typing.Tuple[float, float]]) -> typing.List[int]
:canonical: src.tracking.logging.plotting.charts._calculate_dominance

```{autodoc2-docstring} src.tracking.logging.plotting.charts._calculate_dominance
```
````
