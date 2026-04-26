# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSRRPolicy <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSRRPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HGSRRConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_ruin_and_recreate.policy_hgs_rr.HGSRRPolicy._run_solver
```

````

`````
