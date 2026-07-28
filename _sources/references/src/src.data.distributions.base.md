# {py:mod}`src.data.distributions.base`

```{py:module} src.data.distributions.base
```

```{autodoc2-docstring} src.data.distributions.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseDistribution <src.data.distributions.base.BaseDistribution>`
  - ```{autodoc2-docstring} src.data.distributions.base.BaseDistribution
    :summary:
    ```
````

### API

`````{py:class} BaseDistribution(*args, **kwargs)
:canonical: src.data.distributions.base.BaseDistribution

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.data.distributions.base.BaseDistribution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.data.distributions.base.BaseDistribution.__init__
```

````{py:method} set_sampling_method(sampling_method: str)
:canonical: src.data.distributions.base.BaseDistribution.set_sampling_method

```{autodoc2-docstring} src.data.distributions.base.BaseDistribution.set_sampling_method
```

````

````{py:method} sample(size: typing.Tuple[int, ...], rng: typing.Optional[typing.Union[torch.Generator, numpy.random.Generator]] = None) -> typing.Union[numpy.ndarray, torch.Tensor]
:canonical: src.data.distributions.base.BaseDistribution.sample

```{autodoc2-docstring} src.data.distributions.base.BaseDistribution.sample
```

````

````{py:method} _sample_array(size: typing.Tuple[int, ...], rng: typing.Optional[numpy.random.Generator] = None) -> numpy.ndarray
:canonical: src.data.distributions.base.BaseDistribution._sample_array
:abstractmethod:

```{autodoc2-docstring} src.data.distributions.base.BaseDistribution._sample_array
```

````

````{py:method} _sample_tensor(size: typing.Tuple[int, ...], generator: typing.Optional[torch.Generator] = None) -> torch.Tensor
:canonical: src.data.distributions.base.BaseDistribution._sample_tensor
:abstractmethod:

```{autodoc2-docstring} src.data.distributions.base.BaseDistribution._sample_tensor
```

````

`````
