# {py:mod}`src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh`

```{py:module} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GIHHSolver <src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver
    :summary:
    ```
````

### API

`````{py:class} GIHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.params.GIHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver.__init__
```

````{py:method} solve() -> typing.List[logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution]
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver.solve

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver.solve
```

````

````{py:method} _select_operator() -> str
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._select_operator

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._select_operator
```

````

````{py:method} _apply_selected_operator(current: logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution, operator: str) -> logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_selected_operator

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._apply_selected_operator
```

````

````{py:method} _update_archive(candidate: logic.src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._update_archive

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._update_archive
```

````

````{py:method} _update_episodic_weights() -> None
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._update_episodic_weights

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._update_episodic_weights
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._cost

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.gihh.GIHHSolver._cost
```

````

`````
