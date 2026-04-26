# {py:mod}`src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh`

```{py:module} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SSHHPolicy <src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy
    :summary:
    ```
````

### API

`````{py:class} SSHHPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.ss_hh.SSHHConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.sequence_based_selection_hyper_heuristic.policy_ss_hh.SSHHPolicy._run_solver
```

````

`````
