# {py:mod}`src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga`

```{py:module} src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GAPolicy <src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy
    :summary:
    ```
````

### API

`````{py:class} GAPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ga.GAConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.genetic_algorithm.policy_ga.GAPolicy._run_solver
```

````

`````
