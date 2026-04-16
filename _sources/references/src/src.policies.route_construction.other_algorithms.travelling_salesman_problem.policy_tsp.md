# {py:mod}`src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp`

```{py:module} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TSPPolicy <src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy
    :summary:
    ```
````

### API

`````{py:class} TSPPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.TSPConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.travelling_salesman_problem.policy_tsp.TSPPolicy._run_solver
```

````

`````
