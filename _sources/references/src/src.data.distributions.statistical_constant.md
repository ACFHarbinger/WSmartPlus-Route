# {py:mod}`src.data.distributions.statistical_constant`

```{py:module} src.data.distributions.statistical_constant
```

```{autodoc2-docstring} src.data.distributions.statistical_constant
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Constant <src.data.distributions.statistical_constant.Constant>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_constant.Constant
    :summary:
    ```
````

### API

`````{py:class} Constant(value: typing.Union[float, torch.Tensor] = 1.0)
:canonical: src.data.distributions.statistical_constant.Constant

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.statistical_constant.Constant
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_constant.Constant.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.statistical_constant.Constant._sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_constant.Constant._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.statistical_constant.Constant._sample_array

```{autodoc2-docstring} src.data.distributions.statistical_constant.Constant._sample_array
```

````

`````
