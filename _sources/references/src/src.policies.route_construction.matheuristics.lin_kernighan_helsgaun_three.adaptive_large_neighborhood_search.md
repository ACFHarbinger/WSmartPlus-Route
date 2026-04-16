# {py:mod}`src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search`

```{py:module} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LKH3_ALNS <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.logger
```

````

`````{py:class} LKH3_ALNS(distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue: float = 1.0, cost_unit: float = 1.0, profit_aware_operators: bool = False, mandatory_nodes: typing.Optional[typing.List[int]] = None, coords: typing.Optional[numpy.ndarray] = None, np_rng: typing.Optional[numpy.random.Generator] = None, rng: typing.Optional[random.Random] = None, seed: int = 42, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, max_pool_size: int = 5, n_original: int = 0, R: float = 1.0, C: float = 1.0, perturb_operator_weights: typing.Optional[typing.List[float]] = None)
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS.__init__
```

````{py:method} solve(max_iterations: int = 100, lkh_trials: int = 500, n_vehicles: int = 3, plateau_limit: int = 10, deep_plateau_limit: int = 30, popmusic_subpath_size: int = 50, popmusic_trials: int = 50, popmusic_max_candidates: int = 5, max_k_opt: int = 5, use_ip_merging: bool = True, subgradient_iterations: int = 50, dynamic_topology_discovery: bool = False, native_prize_collecting: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS.solve

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS.solve
```

````

````{py:method} _initialize_solution(lkh_trials: int, n_vehicles: int, popmusic_subpath_size: int, popmusic_trials: int, popmusic_max_candidates: int, max_k_opt: int, use_ip_merging: bool, subgradient_iterations: int, dynamic_topology_discovery: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._initialize_solution

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._initialize_solution
```

````

````{py:method} _route_nodes(nodes: typing.List[int], lkh_trials: int, n_vehicles: int, popmusic_subpath_size: int, popmusic_trials: int, popmusic_max_candidates: int, max_k_opt: int, use_ip_merging: bool, subgradient_iterations: int = 0, initial_routes: typing.Optional[typing.List[typing.List[int]]] = None, dynamic_topology_discovery: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._route_nodes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._route_nodes
```

````

````{py:method} _optimize_routes(routes: typing.List[typing.List[int]], lkh_trials: int, n_vehicles: int, popmusic_subpath_size: int, popmusic_trials: int, popmusic_max_candidates: int, max_k_opt: int, use_ip_merging: bool, subgradient_iterations: int, dynamic_topology_discovery: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._optimize_routes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._optimize_routes
```

````

````{py:method} _destroy_repair(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._destroy_repair

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._destroy_repair
```

````

````{py:method} _compute_objective(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._compute_objective

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._compute_objective
```

````

````{py:method} _select_destroy_operator() -> typing.Callable[[typing.List[typing.List[int]], int], typing.Tuple[typing.List[typing.List[int]], typing.List[int]]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._select_destroy_operator

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._select_destroy_operator
```

````

````{py:method} _select_repair_operator() -> typing.Callable[[typing.List[typing.List[int]], typing.List[int]], typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._select_repair_operator

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._select_repair_operator
```

````

````{py:method} _wrap_historical_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_historical_removal

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_historical_removal
```

````

````{py:method} _wrap_neighbor_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_neighbor_removal

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_neighbor_removal
```

````

````{py:method} _wrap_sector_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_sector_removal

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_sector_removal
```

````

````{py:method} _wrap_savings_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_savings_insertion

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_savings_insertion
```

````

````{py:method} _wrap_deep_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_deep_insertion

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_deep_insertion
```

````

````{py:method} _wrap_nearest_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_nearest_insertion

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_nearest_insertion
```

````

````{py:method} _wrap_historical_profit_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_historical_profit_removal

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_historical_profit_removal
```

````

````{py:method} _wrap_neighbor_profit_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_neighbor_profit_removal

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_neighbor_profit_removal
```

````

````{py:method} _wrap_sector_profit_removal(routes: typing.List[typing.List[int]], n_remove: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_sector_profit_removal

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_sector_profit_removal
```

````

````{py:method} _wrap_savings_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_savings_profit_insertion

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_savings_profit_insertion
```

````

````{py:method} _wrap_deep_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_deep_profit_insertion

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_deep_profit_insertion
```

````

````{py:method} _wrap_nearest_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_nearest_profit_insertion

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._wrap_nearest_profit_insertion
```

````

````{py:method} _select_worst_routes(routes: typing.List[typing.List[int]], n: int) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._select_worst_routes

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._select_worst_routes
```

````

````{py:method} _perturbation(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._perturbation

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._perturbation
```

````

````{py:method} _update_history(routes: typing.List[typing.List[int]], obj: float) -> None
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._update_history

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._update_history
```

````

````{py:method} _update_elite_pool(routes: typing.List[typing.List[int]]) -> None
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._update_elite_pool

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._update_elite_pool
```

````

````{py:method} _compute_routing_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._compute_routing_cost

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.adaptive_large_neighborhood_search.LKH3_ALNS._compute_routing_cost
```

````

`````
