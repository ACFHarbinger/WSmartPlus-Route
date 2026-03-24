# {py:mod}`src.policies.harmony_search.solver`

```{py:module} src.policies.harmony_search.solver
```

```{autodoc2-docstring} src.policies.harmony_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HSSolver <src.policies.harmony_search.solver.HSSolver>`
  - ```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver
    :summary:
    ```
````

### API

`````{py:class} HSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.harmony_search.params.HSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.harmony_search.solver.HSSolver

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.harmony_search.solver.HSSolver.solve

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver.solve
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.harmony_search.solver.HSSolver._build_random_solution

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver._build_random_solution
```

````

````{py:method} _improvise(hm: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[int]]
:canonical: src.policies.harmony_search.solver.HSSolver._improvise

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver._improvise
```

````

````{py:method} _nearest_unvisited(node: int, unvisited: set) -> typing.List[int]
:canonical: src.policies.harmony_search.solver.HSSolver._nearest_unvisited

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver._nearest_unvisited
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.harmony_search.solver.HSSolver._evaluate

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.harmony_search.solver.HSSolver._cost

```{autodoc2-docstring} src.policies.harmony_search.solver.HSSolver._cost
```

````

`````
