# {py:mod}`src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns`

```{py:module} src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VNSPolicy <src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy
    :summary:
    ```
````

### API

`````{py:class} VNSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.vns.VNSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.variable_neighborhood_search.policy_vns.VNSPolicy._run_solver
```

````

`````
