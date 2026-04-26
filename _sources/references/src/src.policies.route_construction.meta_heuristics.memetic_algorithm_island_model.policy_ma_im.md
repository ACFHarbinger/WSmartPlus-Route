# {py:mod}`src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im`

```{py:module} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmIslandModelPolicy <src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmIslandModelPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ma_im.MemeticAlgorithmIslandModelConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_island_model.policy_ma_im.MemeticAlgorithmIslandModelPolicy._run_solver
```

````

`````
