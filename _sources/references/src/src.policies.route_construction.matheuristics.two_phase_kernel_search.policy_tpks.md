# {py:mod}`src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks`

```{py:module} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TPKSPolicy <src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy
    :summary:
    ```
````

### API

`````{py:class} TPKSPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix, sub_wastes, capacity, revenue, cost_unit, values, mandatory_nodes, **kwargs)
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy._run_solver
```

````

````{py:method} execute(**kwargs) -> typing.Tuple[typing.List[int], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.policy_tpks.TPKSPolicy.execute
```

````

`````
