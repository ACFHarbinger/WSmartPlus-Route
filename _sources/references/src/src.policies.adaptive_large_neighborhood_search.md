# {py:mod}`src.policies.adaptive_large_neighborhood_search`

```{py:module} src.policies.adaptive_large_neighborhood_search
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ALNSSolver <src.policies.adaptive_large_neighborhood_search.ALNSSolver>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_alns <src.policies.adaptive_large_neighborhood_search.run_alns>`
  - ```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.run_alns
    :summary:
    ```
````

### API

`````{py:class} ALNSSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.alns_aux.params.ALNSParams)
:canonical: src.policies.adaptive_large_neighborhood_search.ALNSSolver

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver.__init__
```

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.adaptive_large_neighborhood_search.ALNSSolver.solve

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver.solve
```

````

````{py:method} select_operator(weights: typing.List[float]) -> int
:canonical: src.policies.adaptive_large_neighborhood_search.ALNSSolver.select_operator

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver.select_operator
```

````

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.adaptive_large_neighborhood_search.ALNSSolver.calculate_cost

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver.calculate_cost
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.adaptive_large_neighborhood_search.ALNSSolver.build_initial_solution

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.ALNSSolver.build_initial_solution
```

````

`````

````{py:function} run_alns(dist_matrix, demands, capacity, R, C, values, *args)
:canonical: src.policies.adaptive_large_neighborhood_search.run_alns

```{autodoc2-docstring} src.policies.adaptive_large_neighborhood_search.run_alns
```
````
