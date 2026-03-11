# {py:mod}`src.policies.guided_indicators_hyper_heuristic.solution`

```{py:module} src.policies.guided_indicators_hyper_heuristic.solution
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Solution <src.policies.guided_indicators_hyper_heuristic.solution.Solution>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.Solution
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`create_initial_solution <src.policies.guided_indicators_hyper_heuristic.solution.create_initial_solution>`
  - ```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.create_initial_solution
    :summary:
    ```
````

### API

`````{py:class} Solution(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float)
:canonical: src.policies.guided_indicators_hyper_heuristic.solution.Solution

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.Solution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.Solution.__init__
```

````{py:method} evaluate() -> None
:canonical: src.policies.guided_indicators_hyper_heuristic.solution.Solution.evaluate

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.Solution.evaluate
```

````

````{py:method} copy() -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.solution.Solution.copy

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.Solution.copy
```

````

````{py:method} is_feasible() -> bool
:canonical: src.policies.guided_indicators_hyper_heuristic.solution.Solution.is_feasible

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.Solution.is_feasible
```

````

`````

````{py:function} create_initial_solution(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue: float, cost_unit: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None) -> src.policies.guided_indicators_hyper_heuristic.solution.Solution
:canonical: src.policies.guided_indicators_hyper_heuristic.solution.create_initial_solution

```{autodoc2-docstring} src.policies.guided_indicators_hyper_heuristic.solution.create_initial_solution
```
````
