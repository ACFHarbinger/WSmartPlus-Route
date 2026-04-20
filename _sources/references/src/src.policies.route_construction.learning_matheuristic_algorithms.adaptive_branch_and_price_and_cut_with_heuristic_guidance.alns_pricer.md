# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSMultiPeriodPricer <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer
    :summary:
    ```
````

### API

`````{py:class} ALNSMultiPeriodPricer(exact_pricer: logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver, rng_seed: int = 42)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.__init__
```

````{py:method} scenario_overflow_removal(route_nodes: typing.List[int], scenario_prizes: typing.Dict[int, float], dual_values: typing.Dict[int, float], num_remove: int) -> typing.List[int]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.scenario_overflow_removal

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.scenario_overflow_removal
```

````

````{py:method} scenario_aware_insertion(route_nodes: typing.List[int], unrouted: typing.List[int], scenario_prizes: typing.Dict[int, float], dual_values: typing.Dict[int, float], dist_matrix: numpy.ndarray, num_insert: int) -> typing.List[int]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.scenario_aware_insertion

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.scenario_aware_insertion
```

````

````{py:method} _calculate_reduced_cost(route_nodes: typing.List[int], scenario_prizes: typing.Dict[int, float], dual_values: typing.Dict[int, float], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer._calculate_reduced_cost

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer._calculate_reduced_cost
```

````

````{py:method} solve(dual_values: typing.Dict[int, float], scenario_prizes: typing.Dict[int, float], dist_matrix: numpy.ndarray, initial_routes: typing.List[typing.List[int]], all_nodes: typing.List[int], max_routes: int = 5, rc_tolerance: float = 0.0001, alns_iterations: int = 50, **kwargs: typing.Any) -> typing.Tuple[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.Route], bool]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.solve

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.alns_pricer.ALNSMultiPeriodPricer.solve
```

````

`````
