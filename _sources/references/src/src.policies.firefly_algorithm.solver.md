# {py:mod}`src.policies.firefly_algorithm.solver`

```{py:module} src.policies.firefly_algorithm.solver
```

```{autodoc2-docstring} src.policies.firefly_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FASolver <src.policies.firefly_algorithm.solver.FASolver>`
  - ```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver
    :summary:
    ```
````

### API

`````{py:class} FASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.firefly_algorithm.params.FAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.firefly_algorithm.solver.FASolver

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.firefly_algorithm.solver.FASolver.solve

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver.solve
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.firefly_algorithm.solver.FASolver._build_random_solution

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._build_random_solution
```

````

````{py:method} _swap_distance(routes_a: typing.List[typing.List[int]], routes_b: typing.List[typing.List[int]]) -> int
:canonical: src.policies.firefly_algorithm.solver.FASolver._swap_distance

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._swap_distance
```

````

````{py:method} _edge_set(routes: typing.List[typing.List[int]]) -> set
:canonical: src.policies.firefly_algorithm.solver.FASolver._edge_set
:staticmethod:

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._edge_set
```

````

````{py:method} _attract(dim_routes: typing.List[typing.List[int]], bright_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.firefly_algorithm.solver.FASolver._attract

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._attract
```

````

````{py:method} _best_insertion_cost(node: int, routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.firefly_algorithm.solver.FASolver._best_insertion_cost

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._best_insertion_cost
```

````

````{py:method} _random_walk(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.firefly_algorithm.solver.FASolver._random_walk

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._random_walk
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.firefly_algorithm.solver.FASolver._evaluate

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.firefly_algorithm.solver.FASolver._cost

```{autodoc2-docstring} src.policies.firefly_algorithm.solver.FASolver._cost
```

````

`````
