# {py:mod}`src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp`

```{py:module} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSSolverMP <src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP
    :summary:
    ```
````

### API

`````{py:class} ALNSSolverMP(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.params.ALNSParams, horizon: int = 7, stockout_penalty: float = 500.0, forward_looking_depth: int = 3, shift_direction: str = 'both', inventory_lambda: float = 1.0, inter_period_weight: float = 1.0, inter_period_operators: bool = True, mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None)
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP

Bases: {py:obj}`src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns.ALNSSolver`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.__init__
```

````{py:method} _append_inter_period_destroy_ops() -> None
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._append_inter_period_destroy_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._append_inter_period_destroy_ops
```

````

````{py:method} _make_fl_repair() -> typing.Callable
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._make_fl_repair

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._make_fl_repair
```

````

````{py:method} build_initial_horizon_solution() -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.build_initial_horizon_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.build_initial_horizon_solution
```

````

````{py:method} calculate_horizon_cost(horizon_routes: typing.List[typing.List[typing.List[int]]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.calculate_horizon_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.calculate_horizon_cost
```

````

````{py:method} calculate_horizon_profit(horizon_routes: typing.List[typing.List[typing.List[int]]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.calculate_horizon_profit

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.calculate_horizon_profit
```

````

````{py:method} solve_horizon(initial_horizon_routes: typing.Optional[typing.List[typing.List[typing.List[int]]]] = None, scenario_tree: typing.Optional[typing.Any] = None) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.solve_horizon

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP.solve_horizon
```

````

````{py:method} _apply_single_day_ops(day_routes: typing.List[typing.List[int]], d_idx: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._apply_single_day_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._apply_single_day_ops
```

````

````{py:method} _check_accept(current_profit: float, new_profit: float, best_profit: float, iteration: int) -> typing.Tuple[bool, typing.Any]
:canonical: src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._check_accept

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.adaptive_large_neighborhood_search.alns_mp.ALNSSolverMP._check_accept
```

````

`````
