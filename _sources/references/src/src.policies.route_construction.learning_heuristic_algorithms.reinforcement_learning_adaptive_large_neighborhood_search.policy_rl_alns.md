# {py:mod}`src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns`

```{py:module} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLALNSPolicy <src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy
    :summary:
    ```
````

### API

`````{py:class} RLALNSPolicy(config: typing.Any = None)
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy.__init__
```

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.policy_rl_alns.RLALNSPolicy._run_solver
```

````

`````
