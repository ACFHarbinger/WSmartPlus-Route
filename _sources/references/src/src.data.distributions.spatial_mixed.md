# {py:mod}`src.data.distributions.spatial_mixed`

```{py:module} src.data.distributions.spatial_mixed
```

```{autodoc2-docstring} src.data.distributions.spatial_mixed
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Mixed <src.data.distributions.spatial_mixed.Mixed>`
  - ```{autodoc2-docstring} src.data.distributions.spatial_mixed.Mixed
    :summary:
    ```
````

### API

`````{py:class} Mixed(n_cluster_mix: int = 1)
:canonical: src.data.distributions.spatial_mixed.Mixed

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.spatial_mixed.Mixed
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.spatial_mixed.Mixed.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, int, int], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_mixed.Mixed._sample_tensor

```{autodoc2-docstring} src.data.distributions.spatial_mixed.Mixed._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, int, int], rng: typing.Optional[numpy.random.RandomState] = None) -> numpy.ndarray
:canonical: src.data.distributions.spatial_mixed.Mixed._sample_array

```{autodoc2-docstring} src.data.distributions.spatial_mixed.Mixed._sample_array
```

````

`````
