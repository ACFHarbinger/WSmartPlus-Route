# {py:mod}`src.data.distributions.spatial_mix`

```{py:module} src.data.distributions.spatial_mix
```

```{autodoc2-docstring} src.data.distributions.spatial_mix
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MixDistribution <src.data.distributions.spatial_mix.MixDistribution>`
  - ```{autodoc2-docstring} src.data.distributions.spatial_mix.MixDistribution
    :summary:
    ```
````

### API

`````{py:class} MixDistribution(n_cluster: int = 3, n_cluster_mix: int = 1)
:canonical: src.data.distributions.spatial_mix.MixDistribution

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.spatial_mix.MixDistribution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.spatial_mix.MixDistribution.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, int, int], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_mix.MixDistribution._sample_tensor

```{autodoc2-docstring} src.data.distributions.spatial_mix.MixDistribution._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, int, int], rng: typing.Optional[numpy.random.RandomState] = None) -> numpy.ndarray
:canonical: src.data.distributions.spatial_mix.MixDistribution._sample_array

```{autodoc2-docstring} src.data.distributions.spatial_mix.MixDistribution._sample_array
```

````

`````
