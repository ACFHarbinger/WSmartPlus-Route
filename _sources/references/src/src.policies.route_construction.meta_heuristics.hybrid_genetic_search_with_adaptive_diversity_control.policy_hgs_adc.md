# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyHGSADC <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC
    :summary:
    ```
````

### API

`````{py:class} PolicyHGSADC(config: typing.Any = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC

Bases: {py:obj}`logic.src.policies.route_construction.base.base_multi_period_policy.BaseMultiPeriodRoutingPolicy`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC.__init__
```

````{py:method} _run_multi_period_solver(problem: logic.src.interfaces.context.problem_context.ProblemContext, multi_day_ctx: typing.Optional[logic.src.interfaces.context.multi_day_context.MultiDayContext]) -> typing.Tuple[logic.src.interfaces.context.solution_context.SolutionContext, typing.List[typing.List[typing.List[int]]], typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC._run_multi_period_solver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC._run_multi_period_solver
```

````

````{py:method} _construct_multi_period_routes(T: int, base_wastes: numpy.ndarray, daily_increments: numpy.ndarray, dist_matrix: numpy.ndarray, capacity: float, **kwargs: typing.Any) -> typing.Optional[src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC._construct_multi_period_routes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC._construct_multi_period_routes
```

````

````{py:method} _evaluate_individual(ind: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.individual.Individual, base_wastes: numpy.ndarray, daily_increments: numpy.ndarray, dist: numpy.ndarray, capacity: float, n_vehicles: int, T: int) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC._evaluate_individual

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC._evaluate_individual
```

````

````{py:method} run_intra_day_local_search(routes: typing.List[typing.List[typing.List[int]]], dist: numpy.ndarray, capacity: float, loads: numpy.ndarray) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC.run_intra_day_local_search
:staticmethod:

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_with_adaptive_diversity_control.policy_hgs_adc.PolicyHGSADC.run_intra_day_local_search
```

````

`````
