# {py:mod}`src.envs.generators.pdp`

```{py:module} src.envs.generators.pdp
```

```{autodoc2-docstring} src.envs.generators.pdp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PDPGenerator <src.envs.generators.pdp.PDPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.pdp.PDPGenerator
    :summary:
    ```
````

### API

`````{py:class} PDPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', depot_type: str = 'center', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.pdp.PDPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.pdp.PDPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.pdp.PDPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.pdp.PDPGenerator._generate

```{autodoc2-docstring} src.envs.generators.pdp.PDPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.pdp.PDPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.pdp.PDPGenerator._generate_depot
```

````

`````
