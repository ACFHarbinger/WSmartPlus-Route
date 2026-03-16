# {py:mod}`src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns`

```{py:module} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSPolicy <src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy
    :summary:
    ```
````

### API

`````{py:class} HGSALNSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.HGSALNSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy

Bases: {py:obj}`logic.src.policies.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy._get_config_key

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy._run_solver

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.policy_hgs_alns.HGSALNSPolicy._run_solver
```

````

`````
