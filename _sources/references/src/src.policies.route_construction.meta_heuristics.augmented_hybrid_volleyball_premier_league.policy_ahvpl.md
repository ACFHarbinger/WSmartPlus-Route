# {py:mod}`src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl`

```{py:module} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AHVPLPolicy <src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy
    :summary:
    ```
````

### API

`````{py:class} AHVPLPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ahvpl.AHVPLConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.augmented_hybrid_volleyball_premier_league.policy_ahvpl.AHVPLPolicy._run_solver
```

````

`````
