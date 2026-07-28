# {py:mod}`src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma`

```{py:module} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MAPolicy <src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy
    :summary:
    ```
````

### API

`````{py:class} MAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ma.MAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm.policy_ma.MAPolicy._run_solver
```

````

`````
