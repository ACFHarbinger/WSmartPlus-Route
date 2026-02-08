# {py:mod}`src.policies.slack_induction_by_string_removal.solver`

```{py:module} src.policies.slack_induction_by_string_removal.solver
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRSolver <src.policies.slack_induction_by_string_removal.solver.SISRSolver>`
  - ```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver.SISRSolver
    :summary:
    ```
````

### API

`````{py:class} SISRSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.slack_induction_by_string_removal.params.SISRParams)
:canonical: src.policies.slack_induction_by_string_removal.solver.SISRSolver

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver.SISRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver.SISRSolver.__init__
```

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.slack_induction_by_string_removal.solver.SISRSolver.solve

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver.SISRSolver.solve
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.slack_induction_by_string_removal.solver.SISRSolver._calculate_cost

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver.SISRSolver._calculate_cost
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.slack_induction_by_string_removal.solver.SISRSolver._build_initial_solution

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.solver.SISRSolver._build_initial_solution
```

````

`````
