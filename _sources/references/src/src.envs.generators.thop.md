# {py:mod}`src.envs.generators.thop`

```{py:module} src.envs.generators.thop
```

```{autodoc2-docstring} src.envs.generators.thop
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThOPGenerator <src.envs.generators.thop.ThOPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.thop.ThOPGenerator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAX_TIMES <src.envs.generators.thop.MAX_TIMES>`
  - ```{autodoc2-docstring} src.envs.generators.thop.MAX_TIMES
    :summary:
    ```
````

### API

````{py:data} MAX_TIMES
:canonical: src.envs.generators.thop.MAX_TIMES
:type: dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.envs.generators.thop.MAX_TIMES
```

````

`````{py:class} ThOPGenerator(num_loc: int = 20, num_items_per_city: int = 3, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_weight: float = 0.0, max_weight: float = 1.0, min_profit: float = 0.0, max_profit: float = 1.0, capacity: typing.Optional[float] = None, max_time: typing.Optional[float] = None, v_max: float = 1.0, v_min: float = 0.1, depot_type: str = 'random', device: typing.Union[str, torch.device] = 'cpu', rng: typing.Optional[typing.Any] = None, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.envs.generators.thop.ThOPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.thop.ThOPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.thop.ThOPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.thop.ThOPGenerator._generate

```{autodoc2-docstring} src.envs.generators.thop.ThOPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.thop.ThOPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.thop.ThOPGenerator._generate_depot
```

````

`````
