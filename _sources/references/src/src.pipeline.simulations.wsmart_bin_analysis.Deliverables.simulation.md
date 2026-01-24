# {py:mod}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation`

```{py:module} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GridBase <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase
    :summary:
    ```
* - {py:obj}`Simulation <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`fix10pad <src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.fix10pad>`
  - ```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.fix10pad
    :summary:
    ```
````

### API

`````{py:class} GridBase(ids, data_dir, rate_type, info_ver=None, names=None, same_file=False)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.__init__
```

````{py:method} load_data(ids, data_dir, info_ver=None, names=None, rate_type=None, processed=True, same_file=False) -> tuple[pandas.DataFrame, typing.Union[dict, pandas.DataFrame]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.load_data

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.load_data
```

````

````{py:method} __data_preprocess_same_file(data: pandas.DataFrame) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.__data_preprocess_same_file

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.__data_preprocess_same_file
```

````

````{py:method} cacl_freq_tables() -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.cacl_freq_tables

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.cacl_freq_tables
```

````

````{py:method} get_mean_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_mean_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_mean_rate
```

````

````{py:method} get_var_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_var_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_var_rate
```

````

````{py:method} get_std_rate() -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_std_rate

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_std_rate
```

````

````{py:method} get_datarange() -> tuple[pandas.Timestamp, pandas.Timestamp]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_datarange

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_datarange
```

````

````{py:method} sample(n_samples=1) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.sample

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.sample
```

````

````{py:method} get_values_by_date(date, sample: bool = False) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_values_by_date

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_values_by_date
```

````

````{py:method} ___values_by_date(date: pandas.Timestamp) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.___values_by_date

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.___values_by_date
```

````

````{py:method} values_by_date_range(start: pandas.Timestamp = None, end: pandas.Timestamp = None) -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.values_by_date_range

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.values_by_date_range
```

````

````{py:method} get_info(i: int) -> typing.Union[dict, pandas.DataFrame, pandas.Series]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_info

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_info
```

````

````{py:method} get_num_bins() -> int
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_num_bins

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase.get_num_bins
```

````

`````

`````{py:class} Simulation(sim_type: str, ids: list, data_dir: str, train_split: str = None, start_date: str = None, end_date: str = None, rate_type: str = None, predictQ: bool = False, info_ver: str = None, names: str = None, savefit_name: str = None)
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation

Bases: {py:obj}`src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.GridBase`

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.__init__
```

````{py:method} pre_simulate_rates() -> pandas.DataFrame
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.pre_simulate_rates

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.pre_simulate_rates
```

````

````{py:method} reset_simulation()
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.reset_simulation

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.reset_simulation
```

````

````{py:method} get_current_step() -> tuple[numpy.ndarray, typing.Optional[numpy.ndarray], typing.Optional[numpy.ndarray]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.get_current_step

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.get_current_step
```

````

````{py:method} make_collections(bins_index_list: list[int] = None) -> numpy.ndarray
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.make_collections

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.make_collections
```

````

````{py:method} advance_timestep(date=None) -> tuple[int, typing.Optional[numpy.ndarray], typing.Optional[numpy.ndarray]]
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.advance_timestep

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.Simulation.advance_timestep
```

````

`````

````{py:function} fix10pad(s: pandas.Series) -> pandas.Series
:canonical: src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.fix10pad

```{autodoc2-docstring} src.pipeline.simulations.wsmart_bin_analysis.Deliverables.simulation.fix10pad
```
````
