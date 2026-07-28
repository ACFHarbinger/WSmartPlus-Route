# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SelectionResult <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`build_corrected_revenue <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.build_corrected_revenue>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.build_corrected_revenue
    :summary:
    ```
* - {py:obj}`_wastes_from_revenue <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._wastes_from_revenue>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._wastes_from_revenue
    :summary:
    ```
* - {py:obj}`solve_selection_period <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.solve_selection_period>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.solve_selection_period
    :summary:
    ```
* - {py:obj}`_run_greedy <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._run_greedy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._run_greedy
    :summary:
    ```
* - {py:obj}`generate_lifted_cuts <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_lifted_cuts>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_lifted_cuts
    :summary:
    ```
* - {py:obj}`generate_pareto_cuts <src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_pareto_cuts>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_pareto_cuts
    :summary:
    ```
````

### API

`````{py:class} SelectionResult
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult
```

````{py:attribute} period
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.period
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.period
```

````

````{py:attribute} selection
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.selection
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.selection
```

````

````{py:attribute} tour
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.tour
:type: typing.List[int]
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.tour
```

````

````{py:attribute} lagrangian_objective
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.lagrangian_objective
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.lagrangian_objective
```

````

````{py:attribute} raw_objective
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.raw_objective
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.raw_objective
```

````

````{py:attribute} routing_cost_estimate
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.routing_cost_estimate
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.routing_cost_estimate
```

````

````{py:attribute} lifted_penalties
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.lifted_penalties
:type: typing.Dict[int, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.lifted_penalties
```

````

````{py:attribute} pareto_penalties
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.pareto_penalties
:type: typing.Dict[int, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult.pareto_penalties
```

````

`````

````{py:function} build_corrected_revenue(V: numpy.ndarray, lambdas: numpy.ndarray, insertion_costs: numpy.ndarray, gamma: numpy.ndarray, regret_bias: numpy.ndarray, period: int) -> numpy.ndarray
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.build_corrected_revenue

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.build_corrected_revenue
```
````

````{py:function} _wastes_from_revenue(revenue_eff: numpy.ndarray) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._wastes_from_revenue

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._wastes_from_revenue
```
````

````{py:function} solve_selection_period(*, period: int, dist_matrix: numpy.ndarray, revenue_eff: numpy.ndarray, capacity: float, routing_cost_unit: float, mandatory_nodes: typing.List[int], tpks_params: logic.src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams, hard_fix_bins: typing.Optional[typing.List[int]] = None, engine: str = 'tpks', prior_cuts: typing.Optional[typing.Dict[int, float]] = None) -> src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.solve_selection_period

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.solve_selection_period
```
````

````{py:function} _run_greedy(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int]) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._run_greedy

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection._run_greedy
```
````

````{py:function} generate_lifted_cuts(result: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult, dist_matrix: numpy.ndarray, revenue_eff: numpy.ndarray) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_lifted_cuts

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_lifted_cuts
```
````

````{py:function} generate_pareto_cuts(results: typing.List[src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.SelectionResult], dist_matrix: numpy.ndarray, revenue_eff: numpy.ndarray) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_pareto_cuts

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.concurrent_adaptive_lagrangian_matheuristic.selection.generate_pareto_cuts
```
````
