# {py:mod}`src.policies.hulk_hyper_heuristic.hulk`

```{py:module} src.policies.hulk_hyper_heuristic.hulk
```

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKSolver <src.policies.hulk_hyper_heuristic.hulk.HULKSolver>`
  - ```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver
    :summary:
    ```
````

### API

`````{py:class} HULKSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hulk_hyper_heuristic.params.HULKParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None, evaluator=None)
:canonical: src.policies.hulk_hyper_heuristic.hulk.HULKSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver.__init__
```

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hulk_hyper_heuristic.hulk.HULKSolver.solve

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver.solve
```

````

````{py:method} _apply_operators(solution: src.policies.hulk_hyper_heuristic.solution.Solution) -> typing.Tuple[src.policies.hulk_hyper_heuristic.solution.Solution, str, str, typing.Optional[str]]
:canonical: src.policies.hulk_hyper_heuristic.hulk.HULKSolver._apply_operators

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver._apply_operators
```

````

````{py:method} _calculate_removal_size(solution: src.policies.hulk_hyper_heuristic.solution.Solution) -> int
:canonical: src.policies.hulk_hyper_heuristic.hulk.HULKSolver._calculate_removal_size

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver._calculate_removal_size
```

````

````{py:method} _accept_solution(current: src.policies.hulk_hyper_heuristic.solution.Solution, neighbor: src.policies.hulk_hyper_heuristic.solution.Solution) -> typing.Tuple[bool, float]
:canonical: src.policies.hulk_hyper_heuristic.hulk.HULKSolver._accept_solution

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver._accept_solution
```

````

````{py:method} get_operator_statistics() -> typing.Dict
:canonical: src.policies.hulk_hyper_heuristic.hulk.HULKSolver.get_operator_statistics

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.hulk.HULKSolver.get_operator_statistics
```

````

`````
