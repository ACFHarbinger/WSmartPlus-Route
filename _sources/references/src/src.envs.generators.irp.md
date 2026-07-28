# {py:mod}`src.envs.generators.irp`

```{py:module} src.envs.generators.irp
```

```{autodoc2-docstring} src.envs.generators.irp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IRPGenerator <src.envs.generators.irp.IRPGenerator>`
  - ```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator
    :summary:
    ```
````

### API

`````{py:class} IRPGenerator(num_loc: int = 20, num_periods: int = 5, min_loc: float = 0.0, max_loc: float = 1.0, loc_distribution: typing.Union[str, typing.Callable] = 'uniform', vehicle_capacity: float = 1.0, min_demand: float = 0.05, max_demand: float = 0.2, demand_distribution: str = 'uniform', min_holding_cost: float = 0.1, max_holding_cost: float = 1.0, min_init_inventory: float = 0.0, max_init_inventory: float = 0.5, node_inventory_capacity: float = 1.0, depot_type: str = 'corner', device: typing.Union[str, torch.device] = 'cpu', rng: typing.Optional[typing.Any] = None, generator: typing.Optional[torch.Generator] = None, **kwargs: typing.Any)
:canonical: src.envs.generators.irp.IRPGenerator

Bases: {py:obj}`src.envs.generators.base.Generator`

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator.__init__
```

````{py:method} _generate(batch_size: tuple[int, ...]) -> tensordict.TensorDict
:canonical: src.envs.generators.irp.IRPGenerator._generate

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator._generate
```

````

````{py:method} _generate_depot(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.irp.IRPGenerator._generate_depot

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator._generate_depot
```

````

````{py:method} _generate_demands(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.irp.IRPGenerator._generate_demands

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator._generate_demands
```

````

````{py:method} _generate_holding_costs(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.irp.IRPGenerator._generate_holding_costs

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator._generate_holding_costs
```

````

````{py:method} _generate_initial_inventory(batch_size: tuple[int, ...]) -> torch.Tensor
:canonical: src.envs.generators.irp.IRPGenerator._generate_initial_inventory

```{autodoc2-docstring} src.envs.generators.irp.IRPGenerator._generate_initial_inventory
```

````

`````
