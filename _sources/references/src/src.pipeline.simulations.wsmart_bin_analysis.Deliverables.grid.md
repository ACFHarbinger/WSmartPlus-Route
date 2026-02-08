# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GridBase <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fix10pad <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.fix10pad>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.fix10pad
    :summary:
    ```
````

### API

`````{py:class} GridBase(ids, data_dir, rate_type, info_ver=None, names=None, same_file=False)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.__init__
```

````{py:method} load_data(ids, data_dir, info_ver=None, names=None, rate_type=None, processed=True, same_file=False) -> tuple[pandas.DataFrame, typing.Union[dict, pandas.DataFrame]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.load_data

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.load_data
```

````

````{py:method} __data_preprocess_same_file(data: pandas.DataFrame) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.__data_preprocess_same_file

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.__data_preprocess_same_file
```

````

````{py:method} cacl_freq_tables() -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.cacl_freq_tables

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.cacl_freq_tables
```

````

````{py:method} get_mean_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_mean_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_mean_rate
```

````

````{py:method} get_var_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_var_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_var_rate
```

````

````{py:method} get_std_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_std_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_std_rate
```

````

````{py:method} get_datarange() -> tuple[pandas.Timestamp, pandas.Timestamp]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_datarange

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_datarange
```

````

````{py:method} sample(n_samples=1) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.sample

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.sample
```

````

````{py:method} get_values_by_date(date, sample: bool = False) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_values_by_date

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_values_by_date
```

````

````{py:method} ___values_by_date(date: pandas.Timestamp) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.___values_by_date

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.___values_by_date
```

````

````{py:method} values_by_date_range(start: pandas.Timestamp = None, end: pandas.Timestamp = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.values_by_date_range

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.values_by_date_range
```

````

````{py:method} get_info(i: int) -> typing.Union[dict, pandas.DataFrame, pandas.Series]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_info

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_info
```

````

````{py:method} get_num_bins() -> int
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_num_bins

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.GridBase.get_num_bins
```

````

`````

````{py:function} fix10pad(s: pandas.Series) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.fix10pad

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.grid.fix10pad
```
````
