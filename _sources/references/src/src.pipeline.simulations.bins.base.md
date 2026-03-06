# {py:mod}`src.pipeline.simulations.bins.base`

```{py:module} src.pipeline.simulations.bins.base
```

```{autodoc2-docstring} src.pipeline.simulations.bins.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Bins <src.pipeline.simulations.bins.base.Bins>`
  - ```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins
    :summary:
    ```
````

### API

`````{py:class} Bins(n: int, data_dir: str, sample_dist: str = 'gamma', grid: typing.Optional[src.pipeline.simulations.wsmart_bin_analysis.GridBase] = None, area: typing.Optional[str] = None, waste_type: typing.Optional[str] = None, indices: typing.Optional[typing.Union[numpy.ndarray, typing.List[int]]] = None, waste_file: typing.Optional[str] = None, noise_mean: float = 0.0, noise_variance: float = 0.0, n_days: int = 31, n_samples: int = 1, seed: typing.Optional[int] = None)
:canonical: src.pipeline.simulations.bins.base.Bins

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.__init__
```

````{py:method} __get_stdev()
:canonical: src.pipeline.simulations.bins.base.Bins.__get_stdev

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.__get_stdev
```

````

````{py:method} set_statistics(stats_file: str) -> None
:canonical: src.pipeline.simulations.bins.base.Bins.set_statistics

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.set_statistics
```

````

````{py:method} is_stochastic() -> bool
:canonical: src.pipeline.simulations.bins.base.Bins.is_stochastic

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.is_stochastic
```

````

````{py:method} get_fill_history(device: typing.Optional[torch.device] = None) -> typing.Union[numpy.ndarray, torch.Tensor]
:canonical: src.pipeline.simulations.bins.base.Bins.get_fill_history

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.get_fill_history
```

````

````{py:method} get_level_history(device: typing.Optional[torch.device] = None) -> typing.Union[numpy.ndarray, torch.Tensor]
:canonical: src.pipeline.simulations.bins.base.Bins.get_level_history

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.get_level_history
```

````

````{py:method} predict_days_to_overflow(cl: float) -> numpy.ndarray
:canonical: src.pipeline.simulations.bins.base.Bins.predict_days_to_overflow

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.predict_days_to_overflow
```

````

````{py:method} set_indices(indices: typing.Optional[typing.Union[typing.List[int], numpy.ndarray]] = None) -> None
:canonical: src.pipeline.simulations.bins.base.Bins.set_indices

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.set_indices
```

````

````{py:method} set_sample_waste(sample_id: int) -> None
:canonical: src.pipeline.simulations.bins.base.Bins.set_sample_waste

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.set_sample_waste
```

````

````{py:method} collect(idsfull: typing.List[int], cost: float = 0) -> typing.Tuple[numpy.ndarray, float, int, float]
:canonical: src.pipeline.simulations.bins.base.Bins.collect

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.collect
```

````

````{py:method} _process_filling(todaysfilling: numpy.ndarray, noisyfilling: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[int, numpy.ndarray, numpy.ndarray, float]
:canonical: src.pipeline.simulations.bins.base.Bins._process_filling

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins._process_filling
```

````

````{py:method} deterministic_filling(date)
:canonical: src.pipeline.simulations.bins.base.Bins.deterministic_filling

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.deterministic_filling
```

````

````{py:method} load_filling(day: int) -> typing.Tuple[int, numpy.ndarray, numpy.ndarray, float]
:canonical: src.pipeline.simulations.bins.base.Bins.load_filling

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.load_filling
```

````

````{py:method} __setDistribution(param1, param2)
:canonical: src.pipeline.simulations.bins.base.Bins.__setDistribution

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.__setDistribution
```

````

````{py:method} set_gamma_distribution(option=0)
:canonical: src.pipeline.simulations.bins.base.Bins.set_gamma_distribution

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.set_gamma_distribution
```

````

````{py:method} set_collection_level_and_freq(cf=0.9)
:canonical: src.pipeline.simulations.bins.base.Bins.set_collection_level_and_freq

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.set_collection_level_and_freq
```

````

`````
