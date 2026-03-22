# {py:mod}`src.policies.reactive_tabu_search.solver`

```{py:module} src.policies.reactive_tabu_search.solver
```

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RTSSolver <src.policies.reactive_tabu_search.solver.RTSSolver>`
  - ```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver
    :summary:
    ```
````

### API

`````{py:class} RTSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.reactive_tabu_search.params.RTSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver.solve

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver.solve
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh0

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh1

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh2

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh3

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh4

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh4
```

````

````{py:method} _llh5(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh5

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh5
```

````

````{py:method} _llh6(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._llh6

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._llh6
```

````

````{py:method} _hash_routes(routes: typing.List[typing.List[int]]) -> int
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._hash_routes

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._hash_routes
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._build_initial_solution

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._evaluate

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reactive_tabu_search.solver.RTSSolver._cost

```{autodoc2-docstring} src.policies.reactive_tabu_search.solver.RTSSolver._cost
```

````

`````
