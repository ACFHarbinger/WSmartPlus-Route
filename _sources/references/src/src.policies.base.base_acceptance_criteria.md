# {py:mod}`src.policies.base.base_acceptance_criteria`

```{py:module} src.policies.base.base_acceptance_criteria
```

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BaseAcceptanceSolver <src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver>`
  - ```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver
    :summary:
    ```
````

### API

`````{py:class} BaseAcceptanceSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`, {py:obj}`logic.src.interfaces.adapter.IPolicyAdapter`

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._accept
:abstractmethod:

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._accept
```

````

````{py:method} _update_state(iteration: int)
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._update_state

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._update_state
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver.solve

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver.solve
```

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._record_telemetry

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._record_telemetry
```

````

````{py:method} _llh_random_greedy(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_random_greedy

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_random_greedy
```

````

````{py:method} _llh_worst_regret(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_worst_regret

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_worst_regret
```

````

````{py:method} _llh_cluster_greedy(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_cluster_greedy

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_cluster_greedy
```

````

````{py:method} _llh_worst_greedy(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_worst_greedy

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_worst_greedy
```

````

````{py:method} _llh_random_regret(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_random_regret

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._llh_random_regret
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._build_initial_solution

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._evaluate

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._cost

```{autodoc2-docstring} src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver._cost
```

````

`````
