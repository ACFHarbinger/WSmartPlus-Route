# {py:mod}`src.envs.generators.wcvrp`

```{py:module} src.envs.generators.wcvrp
```

```{autodoc2-docstring} src.envs.generators.wcvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`WCVRPGenerator <src.envs.generators.wcvrp.WCVRPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.wcvrp.WCVRPGenerator
    :summary:
    ```
````

### API

`````{py:class} WCVRPGenerator(num_loc: int = 50, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', min_fill: float = 0.0, max_fill: float = 1.0, fill_distribution: str = 'uniform', capacity: float = 100.0, cost_km: float = 1.0, revenue_kg: float = 0.1625, depot_type: str = 'center', device: typing.Union[str, torch.device] = 'cpu', **kwargs: typing.Any)
:canonical: src.envs.generators.wcvrp.WCVRPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.wcvrp.WCVRPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.wcvrp.WCVRPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.wcvrp.WCVRPGenerator._generate

```{autodoc2-docstring} src.envs.generators.wcvrp.WCVRPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.wcvrp.WCVRPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.wcvrp.WCVRPGenerator._generate_depot
```

````

````{py:method} _generate_fill_levels(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.wcvrp.WCVRPGenerator._generate_fill_levels

```{autodoc2-docstring} src.envs.generators.wcvrp.WCVRPGenerator._generate_fill_levels
```

````

`````
