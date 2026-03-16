# {py:mod}`src.policies.guided_indicators_hyper_heuristic.gihh`

```{py:module} src.policies.guided_indicators_hyper_heuristic.gihh
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHSolver <src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_gihh <src.policies.guided_indicators_hyper_heuristic.gihh.run_gihh>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.run_gihh
    :summary:
    ```
````

### API

`````{py:class} GIHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver.solve

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver.solve
```

````

````{py:method} _select_operator() -> str
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._select_operator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._select_operator
```

````

````{py:method} _apply_operator(solution: src.policies.guided_indicators_hyper_heuristic.solution.Solution, operator: str) -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_operator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_operator
```

````

````{py:method} _apply_move_operator(solution: src.policies.guided_indicators_hyper_heuristic.solution.Solution, operator: str) -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_move_operator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_move_operator
```

````

````{py:method} _apply_perturbation_operator(solution: src.policies.guided_indicators_hyper_heuristic.solution.Solution, operator: str) -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_perturbation_operator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_perturbation_operator
```

````

````{py:method} _accept_solution(current: src.policies.guided_indicators_hyper_heuristic.solution.Solution, neighbor: src.policies.guided_indicators_hyper_heuristic.solution.Solution) -> typing.Tuple[bool, float]
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._accept_solution

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._accept_solution
```

````

````{py:method} _update_indicators(operator: str, improvement: float, elapsed_time: float) -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._update_indicators

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._update_indicators
```

````

````{py:method} _roulette_wheel_selection(scores: typing.Dict[str, float]) -> str
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._roulette_wheel_selection

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._roulette_wheel_selection
```

````

`````

````{py:function} run_gihh(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args)
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.run_gihh

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.run_gihh
```
````
