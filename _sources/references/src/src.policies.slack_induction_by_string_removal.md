# {py:mod}`src.policies.slack_induction_by_string_removal`

```{py:module} src.policies.slack_induction_by_string_removal
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRParams <src.policies.slack_induction_by_string_removal.SISRParams>`
  - ```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRParams
    :summary:
    ```
* - {py:obj}`SISRSolver <src.policies.slack_induction_by_string_removal.SISRSolver>`
  - ```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_sisr <src.policies.slack_induction_by_string_removal.run_sisr>`
  - ```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.run_sisr
    :summary:
    ```
````

### API

````{py:class} SISRParams(time_limit: float = 10.0, max_iterations: int = 1000, start_temp: float = 100.0, cooling_rate: float = 0.995, max_string_len: int = 10, avg_string_len: float = 3.0, blink_rate: float = 0.01, destroy_ratio: float = 0.2)
:canonical: src.policies.slack_induction_by_string_removal.SISRParams

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRParams
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRParams.__init__
```

````

`````{py:class} SISRSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.slack_induction_by_string_removal.SISRParams)
:canonical: src.policies.slack_induction_by_string_removal.SISRSolver

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRSolver.__init__
```

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.slack_induction_by_string_removal.SISRSolver.solve

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRSolver.solve
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.slack_induction_by_string_removal.SISRSolver._calculate_cost

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRSolver._calculate_cost
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.slack_induction_by_string_removal.SISRSolver._build_initial_solution

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.SISRSolver._build_initial_solution
```

````

`````

````{py:function} run_sisr(dist_matrix, demands, capacity, R, C, values, **kwargs)
:canonical: src.policies.slack_induction_by_string_removal.run_sisr

```{autodoc2-docstring} src.policies.slack_induction_by_string_removal.run_sisr
```
````
