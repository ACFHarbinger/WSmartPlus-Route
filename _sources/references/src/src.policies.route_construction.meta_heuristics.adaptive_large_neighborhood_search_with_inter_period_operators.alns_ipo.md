# {py:mod}`src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo`

```{py:module} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSSolverIPO <src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO
    :summary:
    ```
````

### API

`````{py:class} ALNSSolverIPO(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.params.ALNSIPOParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None)
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO

Bases: {py:obj}`logic.src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns.ALNSSolver`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.__init__
```

````{py:method} _append_inter_period_destroy_ops() -> None
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._append_inter_period_destroy_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._append_inter_period_destroy_ops
```

````

````{py:method} _append_inter_period_repair_ops() -> None
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._append_inter_period_repair_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._append_inter_period_repair_ops
```

````

````{py:method} _make_fl_repair() -> typing.Callable
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._make_fl_repair

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._make_fl_repair
```

````

````{py:method} build_initial_horizon_solution() -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.build_initial_horizon_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.build_initial_horizon_solution
```

````

````{py:method} calculate_horizon_cost(horizon_routes: typing.List[typing.List[typing.List[int]]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.calculate_horizon_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.calculate_horizon_cost
```

````

````{py:method} calculate_horizon_profit(horizon_routes: typing.List[typing.List[typing.List[int]]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.calculate_horizon_profit

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.calculate_horizon_profit
```

````

````{py:method} solve_horizon(initial_horizon_routes: typing.Optional[typing.List[typing.List[typing.List[int]]]] = None, scenario_tree: typing.Optional[typing.Any] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.solve_horizon

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO.solve_horizon
```

````

````{py:method} _apply_single_day_ops(day_routes: typing.List[typing.List[int]], d_idx: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._apply_single_day_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._apply_single_day_ops
```

````

````{py:method} _check_accept(current_profit: float, new_profit: float, best_profit: float, iteration: int) -> typing.Tuple[bool, typing.Any]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._check_accept

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search_with_inter_period_operators.alns_ipo.ALNSSolverIPO._check_accept
```

````

`````
