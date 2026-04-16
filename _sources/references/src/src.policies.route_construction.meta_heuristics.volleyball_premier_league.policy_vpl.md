# {py:mod}`src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl`

```{py:module} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VPLPolicy <src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy
    :summary:
    ```
````

### API

`````{py:class} VPLPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.vpl.VPLConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.volleyball_premier_league.policy_vpl.VPLPolicy._run_solver
```

````

`````
