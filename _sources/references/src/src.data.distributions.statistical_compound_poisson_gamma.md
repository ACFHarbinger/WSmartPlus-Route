# {py:mod}`src.data.distributions.statistical_compound_poisson_gamma`

```{py:module} src.data.distributions.statistical_compound_poisson_gamma
```

```{autodoc2-docstring} src.data.distributions.statistical_compound_poisson_gamma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CompoundPoissonGamma <src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma
    :summary:
    ```
````

### API

`````{py:class} CompoundPoissonGamma(lam: typing.Union[float, torch.Tensor] = 1.0, alpha: typing.Union[float, torch.Tensor] = 2.0, theta: typing.Union[float, torch.Tensor] = 2.0)
:canonical: src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma._sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma._sample_array

```{autodoc2-docstring} src.data.distributions.statistical_compound_poisson_gamma.CompoundPoissonGamma._sample_array
```

````

`````
