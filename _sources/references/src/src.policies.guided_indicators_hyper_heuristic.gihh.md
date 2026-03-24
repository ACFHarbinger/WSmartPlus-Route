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

### API

`````{py:class} GIHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.guided_indicators_hyper_heuristic.params.GIHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
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

````{py:method} _apply_operator(current: src.policies.guided_indicators_hyper_heuristic.solution.Solution, iteration: int) -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_operator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_operator
```

````

````{py:method} _apply_perturbation_operator(current: src.policies.guided_indicators_hyper_heuristic.solution.Solution) -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_perturbation_operator

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_perturbation_operator
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._evaluate

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._cost

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.gihh.GIHHSolver._cost
```

````

`````
