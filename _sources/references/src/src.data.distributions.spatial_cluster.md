# {py:mod}`src.data.distributions.spatial_cluster`

```{py:module} src.data.distributions.spatial_cluster
```

```{autodoc2-docstring} src.data.distributions.spatial_cluster
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Cluster <src.data.distributions.spatial_cluster.Cluster>`
  - ```{autodoc2-docstring} src.data.distributions.spatial_cluster.Cluster
    :summary:
    ```
````

### API

`````{py:class} Cluster(n_cluster: int = 3)
:canonical: src.data.distributions.spatial_cluster.Cluster

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.spatial_cluster.Cluster
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.spatial_cluster.Cluster.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_cluster.Cluster._sample_tensor

```{autodoc2-docstring} src.data.distributions.spatial_cluster.Cluster._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.spatial_cluster.Cluster._sample_array

```{autodoc2-docstring} src.data.distributions.spatial_cluster.Cluster._sample_array
```

````

`````
