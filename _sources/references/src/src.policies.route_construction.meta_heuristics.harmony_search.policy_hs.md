# {py:mod}`src.policies.route_construction.meta_heuristics.harmony_search.policy_hs`

```{py:module} src.policies.route_construction.meta_heuristics.harmony_search.policy_hs
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.harmony_search.policy_hs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HSPolicy <src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy
    :summary:
    ```
````

### API

`````{py:class} HSPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.hs.HSConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy._config_class
:classmethod:

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy._get_config_key

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.harmony_search.policy_hs.HSPolicy._run_solver
```

````

`````
