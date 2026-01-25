# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TAG <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG
    :summary:
    ```
* - {py:obj}`Container <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container
    :summary:
    ```
````

### API

`````{py:class} TAG
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG

Bases: {py:obj}`enum.Enum`

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG
```

````{py:attribute} LOW_MEASURES
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.LOW_MEASURES
:value: >
   0

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.LOW_MEASURES
```

````

````{py:attribute} INSIDE_BOX
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.INSIDE_BOX
:value: >
   1

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.INSIDE_BOX
```

````

````{py:attribute} OK
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.OK
:value: >
   2

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.OK
```

````

````{py:attribute} WARN
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.WARN
:value: >
   3

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.WARN
```

````

````{py:attribute} LOCAL_WARN
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.LOCAL_WARN
:value: >
   4

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG.LOCAL_WARN
```

````

`````

`````{py:class} Container(my_df: pandas.DataFrame, my_rec: pandas.DataFrame, info: pandas.DataFrame)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.__init__
```

````{py:method} __del__()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.__del__

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.__del__
```

````

````{py:method} get_keys()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_keys

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_keys
```

````

````{py:method} get_vars() -> tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_vars

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_vars
```

````

````{py:method} get_collection_quantities() -> tuple[typing.Optional[numpy.ndarray], typing.Optional[numpy.ndarray]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_collection_quantities

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_collection_quantities
```

````

````{py:method} get_scan_linear_spline(key, interval) -> tuple[pandas.DatetimeIndex, numpy.ndarray]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_scan_linear_spline

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_scan_linear_spline
```

````

````{py:method} get_monotonic_mean_rate_error_splines(interval)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_monotonic_mean_rate_error_splines

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_monotonic_mean_rate_error_splines
```

````

````{py:method} get_monotonic_mean_rate(freq) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_monotonic_mean_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_monotonic_mean_rate
```

````

````{py:method} get_crude_rate(freq) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_crude_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_crude_rate
```

````

````{py:method} get_tag(window: int, mv_thresh: int, min_days: int, use: str) -> src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_tag

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_tag
```

````

````{py:method} get_collections_std()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_collections_std

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.get_collections_std
```

````

````{py:method} set_tag(tag: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.TAG)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.set_tag

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.set_tag
```

````

````{py:method} mark_collections()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.mark_collections

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.mark_collections
```

````

````{py:method} calc_spearman(start_idx: int = 0, end_idx: int = -1)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.calc_spearman

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.calc_spearman
```

````

````{py:method} calc_avg_dist_metric(start_idx: int = 0, end_idx: int = -1)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.calc_avg_dist_metric

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.calc_avg_dist_metric
```

````

````{py:method} calc_max_min_mean(start_idx: int = 0, end_idx: int = -1)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.calc_max_min_mean

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.calc_max_min_mean
```

````

````{py:method} adjust_collections(dist_thresh: int, c_trash: int, max_fill: int)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.adjust_collections

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.adjust_collections
```

````

````{py:method} adjust_one_collection(idx: int, c_trash: int, max_fill: int) -> int
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.adjust_one_collection

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.adjust_one_collection
```

````

````{py:method} place_collections(dist_thresh: int, c_trash: int, max_fill: int, spear_thresh=None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.place_collections

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.place_collections
```

````

````{py:method} place_one_collection(idx: int, c_trash: int, max_fill: int)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.place_one_collection

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.place_one_collection
```

````

````{py:method} clean_box(window: int, mv_thresh: int, use: str)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.clean_box

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.clean_box
```

````

````{py:method} plot_fill(start_date: datetime.datetime, end_date: datetime.datetime, fig_size: tuple = (9, 6))
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.plot_fill

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.plot_fill
```

````

````{py:method} plot_max_min(start_date: datetime.datetime, end_date: datetime.datetime, fig_size: tuple = (9, 6))
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.plot_max_min

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.plot_max_min
```

````

````{py:method} plot_collection_metrics(start_date: datetime.datetime, end_date: datetime.datetime, fig_size: tuple = (9, 6))
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.plot_collection_metrics

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.container.Container.plot_collection_metrics
```

````

`````
