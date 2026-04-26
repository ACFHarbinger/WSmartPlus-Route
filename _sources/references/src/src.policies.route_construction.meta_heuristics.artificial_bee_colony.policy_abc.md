# {py:mod}`src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc`

```{py:module} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ABCPolicy <src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy
    :summary:
    ```
````

### API

`````{py:class} ABCPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.abc.ABCConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.artificial_bee_colony.policy_abc.ABCPolicy._run_solver
```

````

`````
