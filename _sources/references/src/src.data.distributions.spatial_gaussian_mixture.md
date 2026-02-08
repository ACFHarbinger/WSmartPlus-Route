# {py:mod}`src.data.distributions.spatial_gaussian_mixture`

```{py:module} src.data.distributions.spatial_gaussian_mixture
```

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Gaussian_Mixture <src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture>`
  - ```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture
    :summary:
    ```
````

### API

`````{py:class} Gaussian_Mixture(num_modes: int = 0, cdist: int = 0)
:canonical: src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture.__init__
```

````{py:method} sample(size: typing.Tuple[int, int, int]) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture.sample

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture.sample
```

````

````{py:method} _generate_gaussian_mixture(num_loc: int) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._generate_gaussian_mixture

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._generate_gaussian_mixture
```

````

````{py:method} _generate_gaussian(batch_size: int, num_loc: int) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._generate_gaussian

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._generate_gaussian
```

````

````{py:method} _global_min_max_scaling(coords: torch.Tensor) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._global_min_max_scaling

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._global_min_max_scaling
```

````

````{py:method} _batch_normalize_and_center(coords: torch.Tensor) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._batch_normalize_and_center

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.Gaussian_Mixture._batch_normalize_and_center
```

````

`````
