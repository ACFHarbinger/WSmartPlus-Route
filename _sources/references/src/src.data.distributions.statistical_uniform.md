# {py:mod}`src.data.distributions.statistical_uniform`

```{py:module} src.data.distributions.statistical_uniform
```

```{autodoc2-docstring} src.data.distributions.statistical_uniform
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Uniform <src.data.distributions.statistical_uniform.Uniform>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_uniform.Uniform
    :summary:
    ```
````

### API

`````{py:class} Uniform(low: int = 0, high: int = 100)
:canonical: src.data.distributions.statistical_uniform.Uniform

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.statistical_uniform.Uniform
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_uniform.Uniform.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.statistical_uniform.Uniform._sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_uniform.Uniform._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.statistical_uniform.Uniform._sample_array

```{autodoc2-docstring} src.data.distributions.statistical_uniform.Uniform._sample_array
```

````

`````
