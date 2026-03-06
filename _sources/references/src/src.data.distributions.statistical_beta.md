# {py:mod}`src.data.distributions.statistical_beta`

```{py:module} src.data.distributions.statistical_beta
```

```{autodoc2-docstring} src.data.distributions.statistical_beta
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Beta <src.data.distributions.statistical_beta.Beta>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_beta.Beta
    :summary:
    ```
````

### API

`````{py:class} Beta(alpha: typing.Union[float, torch.Tensor] = 0.5, beta: typing.Union[float, torch.Tensor] = 0.5)
:canonical: src.data.distributions.statistical_beta.Beta

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.statistical_beta.Beta
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_beta.Beta.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.statistical_beta.Beta._sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_beta.Beta._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.RandomState] = None) -> numpy.ndarray
:canonical: src.data.distributions.statistical_beta.Beta._sample_array

```{autodoc2-docstring} src.data.distributions.statistical_beta.Beta._sample_array
```

````

`````
