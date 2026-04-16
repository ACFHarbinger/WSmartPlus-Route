# {py:mod}`src.policies.route_construction.matheuristics.kernel_search.policy_ks`

```{py:module} src.policies.route_construction.matheuristics.kernel_search.policy_ks
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KernelSearchPolicy <src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy
    :summary:
    ```
````

### API

`````{py:class} KernelSearchPolicy(config: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: typing.Any, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, float, typing.Optional[logic.src.policies.context.search_context.SearchContext], typing.Optional[logic.src.policies.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.policy_ks.KernelSearchPolicy.execute
```

````

`````
