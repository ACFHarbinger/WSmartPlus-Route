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

`````{py:class} Bins(n: int, data_dir: str, sample_dist: str = 'gamma', grid: typing.Optional[src.pipeline.simulations.wsmart_bin_analysis.GridBase] = None, area: typing.Optional[str] = None, waste_type: typing.Optional[str] = None, indices: typing.Optional[typing.List[int]] = None, waste_file: typing.Optional[str] = None, noise_mean: float = 0.0, noise_variance: float = 0.0)
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

````{py:method} predictdaystooverflow(cl: float) -> numpy.ndarray
:canonical: src.pipeline.simulations.bins.base.Bins.predictdaystooverflow

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.predictdaystooverflow
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

````{py:method} stochasticFilling(n_samples: int = 1, only_fill: bool = False) -> typing.Union[numpy.ndarray, typing.Tuple[int, numpy.ndarray, numpy.ndarray, float]]
:canonical: src.pipeline.simulations.bins.base.Bins.stochasticFilling

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.stochasticFilling
```

````

````{py:method} deterministicFilling(date)
:canonical: src.pipeline.simulations.bins.base.Bins.deterministicFilling

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.deterministicFilling
```

````

````{py:method} loadFilling(day: int) -> typing.Tuple[int, numpy.ndarray, numpy.ndarray, float]
:canonical: src.pipeline.simulations.bins.base.Bins.loadFilling

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.loadFilling
```

````

````{py:method} __setDistribution(param1, param2)
:canonical: src.pipeline.simulations.bins.base.Bins.__setDistribution

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.__setDistribution
```

````

````{py:method} setGammaDistribution(option=0)
:canonical: src.pipeline.simulations.bins.base.Bins.setGammaDistribution

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.setGammaDistribution
```

````

````{py:method} setCollectionLvlandFreq(cf=0.9)
:canonical: src.pipeline.simulations.bins.base.Bins.setCollectionLvlandFreq

```{autodoc2-docstring} src.pipeline.simulations.bins.base.Bins.setCollectionLvlandFreq
```

````

`````
