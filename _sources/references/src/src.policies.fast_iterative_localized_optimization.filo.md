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

````{py:method} _evaluate_routes(routes: typing.List[typing.List[int]]) -> typing.Tuple[float, float]
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver._evaluate_routes

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver._evaluate_routes
```

````

````{py:method} _get_omega(current_routes: typing.List[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver._get_omega

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver._get_omega
```

````

````{py:method} _update_gamma(is_new_best: bool, accepted: bool, ruined: typing.List[int]) -> None
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_gamma

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_gamma
```

````

````{py:method} _update_omega(is_new_best: bool, accepted: bool, ruined: typing.List[int]) -> None
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_omega

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver._update_omega
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.fast_iterative_localized_optimization.filo.FILOSolver.solve

```{autodoc2-docstring} src.policies.fast_iterative_localized_optimization.filo.FILOSolver.solve
```

````

`````
