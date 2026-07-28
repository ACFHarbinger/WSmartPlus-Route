# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TemporalBendersCoordinator <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.logger
```

````

`````{py:class} TemporalBendersCoordinator(tree: typing.Any, prize_engine: typing.Any, capacity: float, revenue: float, cost_unit: float, ph_loop: typing.Optional[typing.Any] = None, alns_pricer: typing.Optional[typing.Any] = None, ml_branching: typing.Optional[typing.Any] = None, scenario_branching: typing.Optional[typing.Any] = None, dive_heuristic: typing.Optional[typing.Any] = None, fix_optimizer: typing.Optional[typing.Any] = None, max_iterations: int = 50, convergence_tol: float = 0.001, cut_pool_max: int = 500, max_visits_per_bin: int = 1, theta_upper_bound: float = 1000000.0, gurobi_master_time_limit: float = 60.0, gurobi_sub_time_limit: float = 30.0, gurobi_mip_gap: float = 0.0001, gurobi_output_flag: bool = False, subproblem_relax: bool = True)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.__init__
```

````{py:method} _add_cut(cut: typing.Dict[str, typing.Any]) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._add_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._add_cut
```

````

````{py:method} solve(**kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.solve

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.solve
```

````

````{py:method} _solve_gurobi(dist_matrix: numpy.ndarray, n_bins: int, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._solve_gurobi

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._solve_gurobi
```

````

````{py:method} _evaluate_subproblems(z_bar: typing.Dict[int, typing.Dict[int, int]], dist_matrix: numpy.ndarray, horizon: int, subproblem: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem) -> typing.Tuple[float, typing.List[typing.List[typing.List[int]]], typing.List[typing.Dict[str, typing.Any]]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._evaluate_subproblems

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._evaluate_subproblems
```

````

````{py:method} _get_scenario_prizes(scenario: typing.Any, fallback_prizes: typing.Dict[int, float], days_remaining: int) -> typing.Dict[int, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._get_scenario_prizes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._get_scenario_prizes
```

````

````{py:method} _estimate_theta_upper_bound(horizon: int) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._estimate_theta_upper_bound

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator._estimate_theta_upper_bound
```

````

````{py:method} generate_benders_cut(day: int, scenario_id: typing.Any, z_bar: typing.Dict[int, int], subproblem_profit: float, subproblem_duals: typing.Dict[int, float], scenario_prob: float) -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.generate_benders_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.temporal_benders.TemporalBendersCoordinator.generate_benders_cut
```

````

`````
