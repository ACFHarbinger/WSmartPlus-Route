# {py:mod}`src.policies.record_to_record_travel.solver`

```{py:module} src.policies.record_to_record_travel.solver
```

```{autodoc2-docstring} src.policies.record_to_record_travel.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RRSolver <src.policies.record_to_record_travel.solver.RRSolver>`
  - ```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver
    :summary:
    ```
````

### API

`````{py:class} RRSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.record_to_record_travel.params.RRParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.record_to_record_travel.solver.RRSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.record_to_record_travel.solver.RRSolver.solve

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver.solve
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.record_to_record_travel.solver.RRSolver._llh0

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.record_to_record_travel.solver.RRSolver._llh1

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.record_to_record_travel.solver.RRSolver._llh2

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.record_to_record_travel.solver.RRSolver._llh3

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.record_to_record_travel.solver.RRSolver._llh4

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.record_to_record_travel.solver.RRSolver._build_initial_solution

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.record_to_record_travel.solver.RRSolver._evaluate

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.record_to_record_travel.solver.RRSolver._cost

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._cost
```

````

`````
