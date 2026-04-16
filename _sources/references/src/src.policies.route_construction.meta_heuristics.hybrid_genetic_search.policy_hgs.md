# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSPolicy <src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HGSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], x_coords: typing.Optional[numpy.ndarray] = None, y_coords: typing.Optional[numpy.ndarray] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search.policy_hgs.HGSPolicy._run_solver
```

````

`````
