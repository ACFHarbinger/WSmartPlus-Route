# {py:mod}`src.policies.fast_iterative_localized_optimization.filo`

```{py:module} src.policies.fast_iterative_localized_optimization.filo
```

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILOSolver <src.policies.fast_iterative_localized_optimization.filo.FILOSolver>`
  - ```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver
    :summary:
    ```
````

### API

`````{py:class} FILOSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.fast_iterative_localized_optimization.params.FILOParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver.__init__
```

````{py:method} calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver.calculate_cost

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver.calculate_cost
```

````

````{py:method} build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver.build_initial_solution

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver.build_initial_solution
```

````

````{py:method} apply_local_search(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver.apply_local_search

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver.apply_local_search
```

````

````{py:method} _update_gamma(is_new_best: bool, max_non_improving: int) -> None
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_gamma

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_gamma
```

````

````{py:method} _update_omega(ruined: typing.List[int], walk_seed: int, ls_cost: float, current_cost: float, shaking_lb: float, shaking_ub: float) -> None
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_omega

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_omega
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver.solve

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver.solve
```

````

`````
