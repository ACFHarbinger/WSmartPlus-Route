# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LagrangianState <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState
    :summary:
    ```
* - {py:obj}`LagrangianCoordinator <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator
    :summary:
    ```
````

### API

`````{py:class} LagrangianState
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState
```

````{py:attribute} x_K
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.x_K
:type: numpy.ndarray
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.x_K
```

````

````{py:attribute} x_R
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.x_R
:type: numpy.ndarray
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.x_R
```

````

````{py:attribute} gamma
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.gamma
:type: numpy.ndarray
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.gamma
```

````

````{py:method} __post_init__() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.__post_init__

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.__post_init__
```

````

````{py:method} snapshot() -> typing.Dict[str, numpy.ndarray]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.snapshot

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState.snapshot
```

````

`````

`````{py:class} LagrangianCoordinator(state: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianState, tracker: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.dual.DualBoundTracker, lag_params: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.params.LagrangianParams)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.__init__
```

````{py:method} set_knapsack_selection(x_K: numpy.ndarray) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.set_knapsack_selection

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.set_knapsack_selection
```

````

````{py:method} set_routing_selection(period: int, x_R_column: numpy.ndarray) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.set_routing_selection

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.set_routing_selection
```

````

````{py:method} submit_period_result(period: int, lagrangian_value_contrib: float, tour_quality_ratio: float, upper_bound: float) -> typing.Tuple[bool, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.submit_period_result

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.submit_period_result
```

````

````{py:method} commit_outer_iteration(full_lagrangian_value: float) -> typing.Dict[str, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.commit_outer_iteration

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.commit_outer_iteration
```

````

````{py:method} reset_outer_iteration() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.reset_outer_iteration

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.reset_outer_iteration
```

````

````{py:method} _clamped_mu(mu: typing.Optional[float] = None) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator._clamped_mu

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator._clamped_mu
```

````

````{py:method} current_dual_bound() -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.current_dual_bound

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.current_dual_bound
```

````

````{py:method} lambdas_snapshot() -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.lambdas_snapshot

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.lambdas_snapshot
```

````

````{py:method} gamma_snapshot() -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.gamma_snapshot

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.lagrangian.LagrangianCoordinator.gamma_snapshot
```

````

`````
