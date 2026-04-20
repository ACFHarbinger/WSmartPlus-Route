# {py:mod}`src.envs.generators.cvrp`

```{py:module} src.envs.generators.cvrp
```

```{autodoc2-docstring} src.envs.generators.cvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPGenerator <src.envs.generators.cvrp.CVRPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.cvrp.CVRPGenerator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CAPACITIES <src.envs.generators.cvrp.CAPACITIES>`
  - ```{autodoc2-docstring} src.envs.generators.cvrp.CAPACITIES
    :summary:
    ```
````

### API

````{py:data} CAPACITIES
:canonical: src.envs.generators.cvrp.CAPACITIES
:type: dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.envs.generators.cvrp.CAPACITIES
```

````

`````{py:class} CVRPGenerator(num_loc: int = 20, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_demand: int = 1, max_demand: int = 10, vehicle_capacity: float = 1.0, capacity: typing.Optional[float] = None, depot_type: str = 'random', device: typing.Union[str, torch.device] = 'cpu', rng: typing.Optional[typing.Any] = None, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.envs.generators.cvrp.CVRPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.cvrp.CVRPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.cvrp.CVRPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.cvrp.CVRPGenerator._generate

```{autodoc2-docstring} src.envs.generators.cvrp.CVRPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.cvrp.CVRPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.cvrp.CVRPGenerator._generate_depot
```

````

`````
