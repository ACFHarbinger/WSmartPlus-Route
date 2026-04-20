# {py:mod}`src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns`

```{py:module} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalBranchingVNSPolicy <src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy
    :summary:
    ```
````

### API

`````{py:class} LocalBranchingVNSPolicy(config: typing.Optional[typing.Dict[str, typing.Any]] = None)
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: typing.Any, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy._run_solver
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]]
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy.execute

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.policy_lb_vns.LocalBranchingVNSPolicy.execute
```

````

`````
