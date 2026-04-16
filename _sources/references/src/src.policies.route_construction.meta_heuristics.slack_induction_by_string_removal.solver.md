# {py:mod}`src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver`

```{py:module} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SISRSolver <src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver
    :summary:
    ```
````

### API

`````{py:class} SISRSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.params.SISRParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver.__init__
```

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver.solve
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._calculate_cost
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._build_initial_solution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._build_initial_solution
```

````

````{py:method} _calculate_profit(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._calculate_profit

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._calculate_profit
```

````

````{py:method} _is_valid_route(route: typing.List[int]) -> bool
:canonical: src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._is_valid_route

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.slack_induction_by_string_removal.solver.SISRSolver._is_valid_route
```

````

`````
