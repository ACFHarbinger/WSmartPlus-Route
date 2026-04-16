# {py:mod}`src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution`

```{py:module} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Solution <src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution>`
  - ```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution
    :summary:
    ```
````

### API

`````{py:class} Solution(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float)
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.__init__
```

````{py:method} evaluate() -> None
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.evaluate

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.evaluate
```

````

````{py:method} copy() -> src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.copy

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.copy
```

````

````{py:method} is_feasible() -> bool
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.is_feasible

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.is_feasible
```

````

````{py:method} is_identical_to(other: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.is_identical_to

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.is_identical_to
```

````

````{py:method} dominates(other: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution) -> bool
:canonical: src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.dominates

```{autodoc2-docstring} src.policies.route_construction.hyper_heuristics.guided_indicators_hyper_heuristic.solution.Solution.dominates
```

````

`````
