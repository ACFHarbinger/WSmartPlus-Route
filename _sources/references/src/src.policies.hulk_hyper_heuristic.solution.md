# {py:mod}`src.policies.hulk_hyper_heuristic.solution`

```{py:module} src.policies.hulk_hyper_heuristic.solution
```

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Solution <src.policies.hulk_hyper_heuristic.solution.Solution>`
  - ```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution
    :summary:
    ```
````

### API

`````{py:class} Solution(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float)
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.__init__
```

````{py:method} _compute_metrics()
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution._compute_metrics

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution._compute_metrics
```

````

````{py:method} calculate_cost() -> float
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.calculate_cost

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.calculate_cost
```

````

````{py:method} calculate_revenue() -> float
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.calculate_revenue

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.calculate_revenue
```

````

````{py:property} cost
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.cost
:type: float

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.cost
```

````

````{py:property} revenue
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.revenue
:type: float

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.revenue
```

````

````{py:property} profit
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.profit
:type: float

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.profit
```

````

````{py:method} is_feasible() -> bool
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.is_feasible

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.is_feasible
```

````

````{py:method} get_route_load(route_idx: int) -> float
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.get_route_load

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.get_route_load
```

````

````{py:method} get_total_nodes() -> int
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.get_total_nodes

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.get_total_nodes
```

````

````{py:method} copy() -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.copy

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.copy
```

````

````{py:method} update_routes(new_routes: typing.List[typing.List[int]])
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.update_routes

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.update_routes
```

````

````{py:method} __repr__() -> str
:canonical: src.policies.hulk_hyper_heuristic.solution.Solution.__repr__

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.solution.Solution.__repr__
```

````

`````
