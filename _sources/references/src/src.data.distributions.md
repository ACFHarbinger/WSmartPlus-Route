# {py:mod}`src.data.distributions`

```{py:module} src.data.distributions
```

```{autodoc2-docstring} src.data.distributions
:allowtitles:
```

## Submodules

```{toctree}
:titlesonly:
:maxdepth: 1

src.data.distributions.spatial_mix
src.data.distributions.spatial_cluster
src.data.distributions.spatial_mixed
src.data.distributions.statistical
src.data.distributions.spatial_mix_multi
src.data.distributions.spatial_gaussian_mixture
src.data.distributions.empirical
```

## Package Contents

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DISTRIBUTION_REGISTRY <src.data.distributions.DISTRIBUTION_REGISTRY>`
  - ```{autodoc2-docstring} src.data.distributions.DISTRIBUTION_REGISTRY
    :summary:
    ```
* - {py:obj}`__all__ <src.data.distributions.__all__>`
  - ```{autodoc2-docstring} src.data.distributions.__all__
    :summary:
    ```
````

### API

````{py:data} DISTRIBUTION_REGISTRY
:canonical: src.data.distributions.DISTRIBUTION_REGISTRY
:type: dict[str, typing.Callable[..., typing.Any]]
:value: >
   None

```{autodoc2-docstring} src.data.distributions.DISTRIBUTION_REGISTRY
```

````

````{py:data} __all__
:canonical: src.data.distributions.__all__
:value: >
   ['Cluster', 'Mixed', 'Gaussian_Mixture', 'Gamma', 'Empirical', 'Mix_Distribution', 'Mix_Multi_Distri...

```{autodoc2-docstring} src.data.distributions.__all__
```

````
