# {py:mod}`src.policies.guided_local_search.solver`

```{py:module} src.policies.guided_local_search.solver
```

```{autodoc2-docstring} src.policies.guided_local_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GLSSolver <src.policies.guided_local_search.solver.GLSSolver>`
  - ```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver
    :summary:
    ```
````

### API

`````{py:class} GLSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.guided_local_search.params.GLSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.guided_local_search.solver.GLSSolver

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.guided_local_search.solver.GLSSolver.solve

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver.solve
```

````

````{py:method} _get_edges(routes: typing.List[typing.List[int]]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._get_edges

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._get_edges
```

````

````{py:method} _update_penalties(routes: typing.List[typing.List[int]]) -> None
:canonical: src.policies.guided_local_search.solver.GLSSolver._update_penalties

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._update_penalties
```

````

````{py:method} _augmented_evaluate(routes: typing.List[typing.List[int]], weight: float) -> float
:canonical: src.policies.guided_local_search.solver.GLSSolver._augmented_evaluate

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._augmented_evaluate
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._llh0

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._llh1

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._llh2

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._llh3

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._llh4

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.guided_local_search.solver.GLSSolver._build_initial_solution

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.guided_local_search.solver.GLSSolver._evaluate

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.guided_local_search.solver.GLSSolver._cost

```{autodoc2-docstring} src.policies.guided_local_search.solver.GLSSolver._cost
```

````

`````
