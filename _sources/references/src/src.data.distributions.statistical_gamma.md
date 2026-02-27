# {py:mod}`src.data.distributions.statistical_gamma`

```{py:module} src.data.distributions.statistical_gamma
```

```{autodoc2-docstring} src.data.distributions.statistical_gamma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Gamma <src.data.distributions.statistical_gamma.Gamma>`
  - ```{autodoc2-docstring} src.data.distributions.statistical_gamma.Gamma
    :summary:
    ```
````

### API

`````{py:class} Gamma(alpha: typing.Union[float, torch.Tensor] = 2.0, theta: typing.Union[float, torch.Tensor] = 2.0, option: typing.Optional[int] = None)
:canonical: src.data.distributions.statistical_gamma.Gamma

```{autodoc2-docstring} src.data.distributions.statistical_gamma.Gamma
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.statistical_gamma.Gamma.__init__
```

````{py:method} _tile_param(size: int, param: typing.List[int]) -> typing.List[int]
:canonical: src.data.distributions.statistical_gamma.Gamma._tile_param

```{autodoc2-docstring} src.data.distributions.statistical_gamma.Gamma._tile_param
```

````

````{py:method} sample_tensor(size: typing.Tuple[int, ...]) -> torch.Tensor
:canonical: src.data.distributions.statistical_gamma.Gamma.sample_tensor

```{autodoc2-docstring} src.data.distributions.statistical_gamma.Gamma.sample_tensor
```

````

````{py:method} sample_array(size: typing.Tuple[int, ...]) -> numpy.ndarray
:canonical: src.data.distributions.statistical_gamma.Gamma.sample_array

```{autodoc2-docstring} src.data.distributions.statistical_gamma.Gamma.sample_array
```

````

`````
