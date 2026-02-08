# {py:mod}`src.envs.generators.base`

```{py:module} src.envs.generators.base
```

```{autodoc2-docstring} src.envs.generators.base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Generator <src.envs.generators.base.Generator>`
  - ```{autodoc2-docstring} src.envs.generators.base.Generator
    :summary:
    ```
````

### API

`````{py:class} Generator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.base.Generator

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.envs.generators.base.Generator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.base.Generator.__init__
```

````{py:property} kwargs
:canonical: src.envs.generators.base.Generator.kwargs
:type: dict[str, typing.Any]

```{autodoc2-docstring} src.envs.generators.base.Generator.kwargs
```

````

````{py:method} to(device: typing.Union[str, torch.device]) -> src.envs.generators.base.Generator
:canonical: src.envs.generators.base.Generator.to

```{autodoc2-docstring} src.envs.generators.base.Generator.to
```

````

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.base.Generator._generate
:abstractmethod:

```{autodoc2-docstring} src.envs.generators.base.Generator._generate
```

````

````{py:method} __call__(batch_size: typing.Union[int, list[int], tuple[int, ...]] = 1) -> tensordict.TensorDict
:canonical: src.envs.generators.base.Generator.__call__

```{autodoc2-docstring} src.envs.generators.base.Generator.__call__
```

````

````{py:method} _generate_locations(batch_size: tuple[int, ...], num_loc: typing.Optional[int] = None) -> torch.Tensor
:canonical: src.envs.generators.base.Generator._generate_locations

```{autodoc2-docstring} src.envs.generators.base.Generator._generate_locations
```

````

````{py:method} _uniform_locations(batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor
:canonical: src.envs.generators.base.Generator._uniform_locations

```{autodoc2-docstring} src.envs.generators.base.Generator._uniform_locations
```

````

````{py:method} _normal_locations(batch_size: tuple[int, ...], num_loc: int) -> torch.Tensor
:canonical: src.envs.generators.base.Generator._normal_locations

```{autodoc2-docstring} src.envs.generators.base.Generator._normal_locations
```

````

````{py:method} _clustered_locations(batch_size: tuple[int, ...], num_loc: int, num_clusters: int = 3) -> torch.Tensor
:canonical: src.envs.generators.base.Generator._clustered_locations

```{autodoc2-docstring} src.envs.generators.base.Generator._clustered_locations
```

````

`````
