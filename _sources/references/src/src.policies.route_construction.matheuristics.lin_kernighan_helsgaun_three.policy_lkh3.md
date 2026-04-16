# {py:mod}`src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3`

```{py:module} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LKH3Policy <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy
    :summary:
    ```
````

### API

`````{py:class} LKH3Policy(config: typing.Optional[typing.Union[logic.src.configs.policies.lkh3.LKH3Config, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._get_config_key
```

````

````{py:method} _tour_to_routes(tour: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._tour_to_routes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._tour_to_routes
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._run_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._run_solver
```

````

````{py:method} _route_cost(routes: typing.List[typing.List[int]], dist: numpy.ndarray) -> float
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._route_cost
:staticmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._route_cost
```

````

````{py:method} _enforce_mandatory_and_filter(routes: typing.List[typing.List[int]], mandatory_set: set, dist: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._enforce_mandatory_and_filter

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.policy_lkh3.LKH3Policy._enforce_mandatory_and_filter
```

````

`````
