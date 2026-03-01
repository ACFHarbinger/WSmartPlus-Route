# {py:mod}`src.policies.iterated_local_search.solver`

```{py:module} src.policies.iterated_local_search.solver
```

```{autodoc2-docstring} src.policies.iterated_local_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ILSSolver <src.policies.iterated_local_search.solver.ILSSolver>`
  - ```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver
    :summary:
    ```
````

### API

`````{py:class} ILSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.iterated_local_search.params.ILSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.iterated_local_search.solver.ILSSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.iterated_local_search.solver.ILSSolver.solve

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver.solve
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._perturb

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._perturb
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._llh0

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._llh1

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._llh2

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._llh3

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._llh4

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.iterated_local_search.solver.ILSSolver._build_initial_solution

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.iterated_local_search.solver.ILSSolver._evaluate

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.iterated_local_search.solver.ILSSolver._cost

```{autodoc2-docstring} src.policies.iterated_local_search.solver.ILSSolver._cost
```

````

`````
