# {py:mod}`src.data.distributions.spatial_distance`

```{py:module} src.data.distributions.spatial_distance
```

```{autodoc2-docstring} src.data.distributions.spatial_distance
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Distance <src.data.distributions.spatial_distance.Distance>`
  - ```{autodoc2-docstring} src.data.distributions.spatial_distance.Distance
    :summary:
    ```
````

### API

`````{py:class} Distance(graph: typing.Tuple[typing.Any, typing.Any])
:canonical: src.data.distributions.spatial_distance.Distance

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.spatial_distance.Distance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.spatial_distance.Distance.__init__
```

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[typing.Union[torch.Generator, numpy.random.RandomState]] = None) -> numpy.ndarray
:canonical: src.data.distributions.spatial_distance.Distance._sample_array

```{autodoc2-docstring} src.data.distributions.spatial_distance.Distance._sample_array
```

````

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_distance.Distance._sample_tensor

```{autodoc2-docstring} src.data.distributions.spatial_distance.Distance._sample_tensor
```

````

`````
