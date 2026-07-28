# {py:mod}`src.data.distributions.statistical_empirical`

```{py:module} src.data.distributions.statistical_empirical
```

```{autodoc2-docstring} src.data.distributions.statistical_empirical
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Empirical <src.data.distributions.statistical_empirical.Empirical>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_empirical.Empirical
    :summary:
    ```
````

### API

`````{py:class} Empirical(grid: typing.Optional[logic.src.pipeline.simulations.wsmart_bin_analysis.GridBase] = None, area: typing.Optional[str] = None, indices: typing.Optional[numpy.ndarray] = None, data_path: typing.Optional[str] = None, dataset: typing.Optional[typing.Any] = None)
:canonical: src.data.distributions.statistical_empirical.Empirical

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.statistical_empirical.Empirical
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_empirical.Empirical.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.statistical_empirical.Empirical._sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_empirical.Empirical._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.statistical_empirical.Empirical._sample_array

```{autodoc2-docstring} src.data.distributions.statistical_empirical.Empirical._sample_array
```

````

`````
