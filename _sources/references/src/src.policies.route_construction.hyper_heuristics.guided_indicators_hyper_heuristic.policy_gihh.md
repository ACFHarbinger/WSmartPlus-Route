# {py:mod}`src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh`

```{py:module} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHPolicy <src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy
    :summary:
    ```
````

### API

`````{py:class} GIHHPolicy(config: typing.Optional[typing.Union[logic.src.configs.policies.GIHHConfig, typing.Dict[str, typing.Any]]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy.__init__
```

````{py:method} _config_class() -> typing.Optional[typing.Type]
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy._get_config_key

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy._get_config_key
```

````

````{py:method} _run_solver(sub_dist_matrix: numpy.ndarray, sub_wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy._run_solver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.policy_gihh.GIHHPolicy._run_solver
```

````

`````
