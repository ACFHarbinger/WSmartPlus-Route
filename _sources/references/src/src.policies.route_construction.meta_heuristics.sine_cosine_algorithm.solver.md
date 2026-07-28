# {py:mod}`src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver`

```{py:module} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCASolver <src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver
    :summary:
    ```
````

### API

`````{py:class} SCASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.params.SCAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver.solve
```

````

````{py:method} _decode(x: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver._decode

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver._decode
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.sine_cosine_algorithm.solver.SCASolver._cost
```

````

`````
