# {py:mod}`src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco`

```{py:module} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AdaptiveRouteConstructorOrchestrator <src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_sigmoid_reward <src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco._sigmoid_reward>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco._sigmoid_reward
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.logger
```

````

`````{py:class} AdaptiveRouteConstructorOrchestrator(config: typing.Any = None)
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator

Bases: {py:obj}`logic.src.policies.route_construction.base.base_routing_policy.BaseRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.__init__
```

````{py:method} _config_class()
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._config_class
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._config_class
```

````

````{py:method} _get_config_key() -> str
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._get_config_key
:classmethod:

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._get_config_key
```

````

````{py:method} _initialize_constructors() -> None
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._initialize_constructors

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._initialize_constructors
```

````

````{py:method} _init_weights() -> None
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._init_weights

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._init_weights
```

````

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.Union[typing.List[int], typing.List[typing.List[int]]], float, float, typing.Optional[logic.src.interfaces.context.search_context.SearchContext], typing.Optional[typing.Any]]
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.execute

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.execute
```

````

````{py:method} _select_sequence() -> typing.List[int]
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._select_sequence

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._select_sequence
```

````

````{py:method} _apply_strategy(scores: numpy.ndarray, available: typing.List[int]) -> int
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._apply_strategy

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._apply_strategy
```

````

````{py:method} _update_weights(sequence: typing.List[int], step_profits: typing.List[float], step_costs: typing.List[float]) -> None
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._update_weights

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._update_weights
```

````

````{py:method} _apply_decay() -> None
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._apply_decay

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._apply_decay
```

````

````{py:method} get_weight_summary() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.get_weight_summary

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.get_weight_summary
```

````

````{py:method} best_sequence() -> typing.List[str]
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.best_sequence

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.best_sequence
```

````

````{py:method} reset_weights() -> None
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.reset_weights

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator.reset_weights
```

````

````{py:method} _run_solver(sub_dist_matrix: typing.Any, sub_wastes: typing.Any, capacity: float, revenue: float, cost_unit: float, values: typing.Any, mandatory_nodes: typing.List[int], **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._run_solver

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco.AdaptiveRouteConstructorOrchestrator._run_solver
```

````

`````

````{py:function} _sigmoid_reward(raw: float, scale: float = 1.0) -> float
:canonical: src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco._sigmoid_reward

```{autodoc2-docstring} src.policies.route_construction.other_algorithms.adaptive_route_constructor_orchestrator.policy_arco._sigmoid_reward
```
````
