# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LNSSolver <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver
    :summary:
    ```
````

### API

`````{py:class} LNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.LNSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.__init__
```

````{py:method} _initialize_solve(initial_solution: typing.Optional[typing.List[typing.List[int]]]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._initialize_solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._initialize_solve
```

````

````{py:method} _select_and_apply_operators(current_routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], int, int]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._select_and_apply_operators

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._select_and_apply_operators
```

````

````{py:method} _accept_solution(current_profit: float, new_profit: float, T: float) -> bool
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._accept_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._accept_solution
```

````

````{py:method} _update_weights(d_idx: int, r_idx: int, score: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._update_weights

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver._update_weights
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.solve
```

````

````{py:method} select_operator(weights: typing.List[float]) -> int
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.select_operator

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.select_operator
```

````

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.calculate_cost
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.build_initial_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.lns.LNSSolver.build_initial_solution
```

````

`````
