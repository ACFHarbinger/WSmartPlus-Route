# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.sample_gen`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.sample_gen
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OldGridBase <src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fix10pad <src.pipeline.simulations.wsmart_bin_analysis.sample_gen.fix10pad>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.fix10pad
    :summary:
    ```
````

### API

`````{py:class} OldGridBase(data_dir: str, area: str)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.__init__
```

````{py:method} __data_preprocess(data: pandas.DataFrame) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.__data_preprocess

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.__data_preprocess
```

````

````{py:method} __calc_freq_tables() -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.__calc_freq_tables

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.__calc_freq_tables
```

````

````{py:method} get_mean_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_mean_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_mean_rate
```

````

````{py:method} get_var_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_var_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_var_rate
```

````

````{py:method} get_std_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_std_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_std_rate
```

````

````{py:method} get_daterange() -> tuple[pandas.Timestamp, pandas.Timestamp]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_daterange

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_daterange
```

````

````{py:method} sample(n_samples=1) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.sample

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.sample
```

````

````{py:method} get_values_by_date(date, sample: bool = False) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_values_by_date

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_values_by_date
```

````

````{py:method} ___values_by_date(date: pandas.Timestamp) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.___values_by_date

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.___values_by_date
```

````

````{py:method} get_info(i: int) -> dict
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_info

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_info
```

````

````{py:method} get_num_bins() -> int
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_num_bins

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.get_num_bins
```

````

````{py:method} load_data(processed=True) -> tuple[pandas.DataFrame, pandas.DataFrame]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.load_data

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.OldGridBase.load_data
```

````

`````

````{py:function} fix10pad(s: pandas.Series) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.sample_gen.fix10pad

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.sample_gen.fix10pad
```
````
