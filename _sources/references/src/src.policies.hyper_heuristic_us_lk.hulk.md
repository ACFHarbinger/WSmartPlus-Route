# {py:mod}`src.policies.hyper_heuristic_us_lk.hulk`

```{py:module} src.policies.hyper_heuristic_us_lk.hulk
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKSolver <src.policies.hyper_heuristic_us_lk.hulk.HULKSolver>`
  - ```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver
    :summary:
    ```
````

### API

`````{py:class} HULKSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hyper_heuristic_us_lk.params.HULKParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, evaluator=None)
:canonical: src.policies.hyper_heuristic_us_lk.hulk.HULKSolver

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver.__init__
```

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hyper_heuristic_us_lk.hulk.HULKSolver.solve

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver.solve
```

````

````{py:method} _apply_operators(solution: src.policies.hyper_heuristic_us_lk.solution.Solution) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, str, str, typing.Optional[str]]
:canonical: src.policies.hyper_heuristic_us_lk.hulk.HULKSolver._apply_operators

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver._apply_operators
```

````

````{py:method} _calculate_removal_size(solution: src.policies.hyper_heuristic_us_lk.solution.Solution) -> int
:canonical: src.policies.hyper_heuristic_us_lk.hulk.HULKSolver._calculate_removal_size

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver._calculate_removal_size
```

````

````{py:method} _accept_solution(current: src.policies.hyper_heuristic_us_lk.solution.Solution, neighbor: src.policies.hyper_heuristic_us_lk.solution.Solution) -> typing.Tuple[bool, float]
:canonical: src.policies.hyper_heuristic_us_lk.hulk.HULKSolver._accept_solution

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver._accept_solution
```

````

````{py:method} get_operator_statistics() -> typing.Dict
:canonical: src.policies.hyper_heuristic_us_lk.hulk.HULKSolver.get_operator_statistics

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.hulk.HULKSolver.get_operator_statistics
```

````

`````
