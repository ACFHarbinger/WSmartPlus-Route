# {py:mod}`src.envs.generators.op`

```{py:module} src.envs.generators.op
```

```{autodoc2-docstring} src.envs.generators.op
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OPGenerator <src.envs.generators.op.OPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.op.OPGenerator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAX_LENGTHS <src.envs.generators.op.MAX_LENGTHS>`
  - ```{autodoc2-docstring} src.envs.generators.op.MAX_LENGTHS
    :summary:
    ```
````

### API

````{py:data} MAX_LENGTHS
:canonical: src.envs.generators.op.MAX_LENGTHS
:type: dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.envs.generators.op.MAX_LENGTHS
```

````

`````{py:class} OPGenerator(num_loc: int = 20, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', prize_type: str = 'dist', max_length: typing.Optional[float] = None, depot_type: str = 'random', device: typing.Union[str, torch.device] = 'cpu', rng: typing.Optional[typing.Any] = None, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.envs.generators.op.OPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.op.OPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.op.OPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.op.OPGenerator._generate

```{autodoc2-docstring} src.envs.generators.op.OPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.op.OPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.op.OPGenerator._generate_depot
```

````

`````
