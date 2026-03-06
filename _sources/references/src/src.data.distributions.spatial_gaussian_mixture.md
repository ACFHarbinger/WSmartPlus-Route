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

* - {py:obj}`GaussianMixture <src.data.distributions.spatial_gaussian_mixture.GaussianMixture>`
  - ```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture
    :summary:
    ```
````

### API

`````{py:class} GaussianMixture(num_modes: int = 0, cdist: int = 0)
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture

Bases: {py:obj}`src.data.distributions.base.BaseDistribution`

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture.__init__
```

````{py:method} _sample_tensor(size: typing.Tuple[int, int, int], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._sample_tensor

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._sample_tensor
```

````

````{py:method} _sample_array(size: typing.Tuple[int, int, int], rng: typing.Optional[numpy.random.RandomState] = None) -> numpy.ndarray
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._sample_array

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._sample_array
```

````

````{py:method} _generate_gaussian_mixture(num_loc: int, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian_mixture

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian_mixture
```

````

````{py:method} _generate_gaussian(batch_size: int, num_loc: int, generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian
```

````

````{py:method} _generate_gaussian_mixture_array(num_loc: int, rng: numpy.random.RandomState) -> numpy.ndarray
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian_mixture_array

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian_mixture_array
```

````

````{py:method} _generate_gaussian_array(batch_size: int, num_loc: int, rng: numpy.random.RandomState) -> numpy.ndarray
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian_array

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._generate_gaussian_array
```

````

````{py:method} _global_min_max_scaling(coords: typing.Union[numpy.ndarray, torch.Tensor]) -> typing.Union[numpy.ndarray, torch.Tensor]
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._global_min_max_scaling

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._global_min_max_scaling
```

````

````{py:method} _batch_normalize_and_center(coords: typing.Union[numpy.ndarray, torch.Tensor]) -> typing.Union[numpy.ndarray, torch.Tensor]
:canonical: src.data.distributions.spatial_gaussian_mixture.GaussianMixture._batch_normalize_and_center

```{autodoc2-docstring} src.data.distributions.spatial_gaussian_mixture.GaussianMixture._batch_normalize_and_center
```

````

`````
