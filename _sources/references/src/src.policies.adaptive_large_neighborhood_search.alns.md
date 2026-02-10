# {py:mod}`src.policies.adaptive_large_neighborhood_search.alns`

```{py:module} src.policies.adaptive_large_neighborhood_search.alns
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSSolver <src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_alns <src.policies.adaptive_large_neighborhood_search.alns.run_alns>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.run_alns
    :summary:
    ```
````

### API

`````{py:class} ALNSSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.adaptive_large_neighborhood_search.params.ALNSParams)
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.__init__
```

````{py:method} _initialize_solve(initial_solution: typing.Optional[typing.List[typing.List[int]]])
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._initialize_solve

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._initialize_solve
```

````

````{py:method} _select_and_apply_operators(current_routes)
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._select_and_apply_operators

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._select_and_apply_operators
```

````

````{py:method} _accept_solution(current_profit, new_profit, T)
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._accept_solution

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._accept_solution
```

````

````{py:method} _update_weights(d_idx, r_idx, score)
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._update_weights

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver._update_weights
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.solve

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.solve
```

````

````{py:method} select_operator(weights: typing.List[float]) -> int
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.select_operator

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.select_operator
```

````

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.calculate_cost

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.calculate_cost
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.build_initial_solution

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.ALNSSolver.build_initial_solution
```

````

`````

````{py:function} run_alns(dist_matrix, demands, capacity, R, C, values, *args)
:canonical: src.policies.adaptive_large_neighborhood_search.alns.run_alns

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.alns.run_alns
```
````
