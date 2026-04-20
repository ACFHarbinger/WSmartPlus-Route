# {py:mod}`src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp`

```{py:module} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CVRPPolicy <src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy
    :summary:
    ```
````

### API

`````{py:class} CVRPPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.CVRPConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.capacitated_vehicle_routing_problem.policy_cvrp.CVRPPolicy.execute
```

````

`````
