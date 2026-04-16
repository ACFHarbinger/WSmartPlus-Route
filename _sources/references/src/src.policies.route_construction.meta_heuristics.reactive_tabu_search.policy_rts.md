# {py:mod}`src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts`

```{py:module} src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RTSPolicy <src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy
    :summary:
    ```
````

### API

`````{py:class} RTSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.rts.RTSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.reactive_tabu_search.policy_rts.RTSPolicy._run_solver
```

````

`````
