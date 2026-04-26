# {py:mod}`src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo`

```{py:module} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILOPolicy <src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy
    :summary:
    ```
````

### API

`````{py:class} FILOPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.FILOConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.fast_iterative_localized_optimization.policy_filo.FILOPolicy._run_solver
```

````

`````
