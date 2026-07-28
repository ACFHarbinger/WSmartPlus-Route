# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSSolver <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver
    :summary:
    ```
````

### API

`````{py:class} ALNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.ALNSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.__init__
```

````{py:method} _build_destroy_ops() -> typing.Tuple[typing.List[typing.Callable], typing.List[str]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._build_destroy_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._build_destroy_ops
```

````

````{py:method} _build_repair_ops() -> typing.Tuple[typing.List[typing.Callable], typing.List[str], typing.List[bool]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._build_repair_ops

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._build_repair_ops
```

````

````{py:method} _get_noise() -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._get_noise

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._get_noise
```

````

````{py:method} _hash_solution(routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.Tuple[int, ...], ...]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._hash_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._hash_solution
```

````

````{py:method} _initialize_solve(initial_solution: typing.Optional[typing.List[typing.List[int]]]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._initialize_solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._initialize_solve
```

````

````{py:method} _select_and_apply_operators(current_routes: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], int, int]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._select_and_apply_operators

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._select_and_apply_operators
```

````

````{py:method} _update_weights(d_idx: int, r_idx: int, score: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._update_weights

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._update_weights
```

````

````{py:method} _end_segment() -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._end_segment

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver._end_segment
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.solve
```

````

````{py:method} select_operator(weights: typing.List[float]) -> int
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.select_operator

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.select_operator
```

````

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.calculate_cost
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.build_initial_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.alns.ALNSSolver.build_initial_solution
```

````

`````
