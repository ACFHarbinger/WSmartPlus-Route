# {py:mod}`src.policies.old_bachelor_acceptance.solver`

```{py:module} src.policies.old_bachelor_acceptance.solver
```

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OBASolver <src.policies.old_bachelor_acceptance.solver.OBASolver>`
  - ```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver
    :summary:
    ```
````

### API

`````{py:class} OBASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.old_bachelor_acceptance.params.OBAParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver.solve

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver.solve
```

````

````{py:method} _llh0(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._llh0

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._llh0
```

````

````{py:method} _llh1(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._llh1

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._llh1
```

````

````{py:method} _llh2(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._llh2

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._llh2
```

````

````{py:method} _llh3(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._llh3

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._llh3
```

````

````{py:method} _llh4(routes: typing.List[typing.List[int]], n: int) -> typing.List[typing.List[int]]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._llh4

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._llh4
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._build_initial_solution

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._build_initial_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._evaluate

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._cost

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._cost
```

````

`````
