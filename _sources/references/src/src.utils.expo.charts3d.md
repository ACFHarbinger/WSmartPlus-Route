# {py:mod}`src.utils.expo.charts3d`

```{py:module} src.utils.expo.charts3d
```

```{autodoc2-docstring} src.utils.expo.charts3d
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`plot_3dchart <src.utils.expo.charts3d.plot_3dchart>`
  - ```{autodoc2-docstring} src.utils.expo.charts3d.plot_3dchart
    :summary:
    ```
* - {py:obj}`_plot_3d_series <src.utils.expo.charts3d._plot_3d_series>`
  - ```{autodoc2-docstring} src.utils.expo.charts3d._plot_3d_series
    :summary:
    ```
* - {py:obj}`_annotate_3d_plot <src.utils.expo.charts3d._annotate_3d_plot>`
  - ```{autodoc2-docstring} src.utils.expo.charts3d._annotate_3d_plot
    :summary:
    ```
* - {py:obj}`_set_3d_plot_attributes <src.utils.expo.charts3d._set_3d_plot_attributes>`
  - ```{autodoc2-docstring} src.utils.expo.charts3d._set_3d_plot_attributes
    :summary:
    ```
* - {py:obj}`_save_3d_plot <src.utils.expo.charts3d._save_3d_plot>`
  - ```{autodoc2-docstring} src.utils.expo.charts3d._save_3d_plot
    :summary:
    ```
````

### API

````{py:function} plot_3dchart(output_dest: str, graph_log: numpy.ndarray, plot_func: typing.Callable[..., typing.Any], policies: typing.List[str], x_label: typing.Optional[str] = None, y_label: typing.Optional[str] = None, z_label: typing.Optional[str] = None, title: typing.Optional[str] = None, fsave: bool = True, scale: str = 'linear', x_values: typing.Optional[typing.Union[typing.List[float], numpy.ndarray]] = None, linestyles: typing.Optional[typing.List[str]] = None, markers: typing.Optional[typing.List[str]] = None, annotate: bool = True) -> None
:canonical: src.utils.expo.charts3d.plot_3dchart

```{autodoc2-docstring} src.utils.expo.charts3d.plot_3dchart
```
````

````{py:function} _plot_3d_series(ax: mpl_toolkits.mplot3d.Axes3D, plot_func: typing.Callable[..., typing.Any], graph_log: numpy.ndarray, policies: typing.List[str], x_values: typing.Optional[typing.Union[typing.List[float], numpy.ndarray]], linestyles: typing.Optional[typing.List[str]], markers: typing.Optional[typing.List[str]]) -> None
:canonical: src.utils.expo.charts3d._plot_3d_series

```{autodoc2-docstring} src.utils.expo.charts3d._plot_3d_series
```
````

````{py:function} _annotate_3d_plot(ax: mpl_toolkits.mplot3d.Axes3D, graph_log: numpy.ndarray, x_values: typing.Optional[typing.Union[typing.List[float], numpy.ndarray]]) -> None
:canonical: src.utils.expo.charts3d._annotate_3d_plot

```{autodoc2-docstring} src.utils.expo.charts3d._annotate_3d_plot
```
````

````{py:function} _set_3d_plot_attributes(ax: mpl_toolkits.mplot3d.Axes3D, scale: str, x_label: typing.Optional[str], y_label: typing.Optional[str], z_label: typing.Optional[str], policies: typing.List[str]) -> None
:canonical: src.utils.expo.charts3d._set_3d_plot_attributes

```{autodoc2-docstring} src.utils.expo.charts3d._set_3d_plot_attributes
```
````

````{py:function} _save_3d_plot(output_dest: str, x_values: typing.Optional[typing.Union[typing.List[float], numpy.ndarray]]) -> None
:canonical: src.utils.expo.charts3d._save_3d_plot

```{autodoc2-docstring} src.utils.expo.charts3d._save_3d_plot
```
````
