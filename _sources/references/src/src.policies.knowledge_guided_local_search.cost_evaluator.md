# {py:mod}`src.policies.knowledge_guided_local_search.cost_evaluator`

```{py:module} src.policies.knowledge_guided_local_search.cost_evaluator
```

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CostEvaluator <src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator>`
  - ```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator
    :summary:
    ```
````

### API

`````{py:class} CostEvaluator(dist_matrix: numpy.ndarray)
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.__init__
```

````{py:method} _compute_baseline_cost() -> float
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator._compute_baseline_cost

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator._compute_baseline_cost
```

````

````{py:method} enable_penalization()
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.enable_penalization

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.enable_penalization
```

````

````{py:method} disable_penalization()
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.disable_penalization

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.disable_penalization
```

````

````{py:method} reset_penalties()
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.reset_penalties

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.reset_penalties
```

````

````{py:method} get_distance_matrix() -> numpy.ndarray
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.get_distance_matrix

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.get_distance_matrix
```

````

````{py:method} _compute_route_center(route: typing.List[int], locations: numpy.ndarray) -> typing.Tuple[float, float]
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator._compute_route_center

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator._compute_route_center
```

````

````{py:method} _compute_edge_width(u: int, v: int, cx: float, cy: float, locations: numpy.ndarray) -> float
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator._compute_edge_width

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator._compute_edge_width
```

````

````{py:method} evaluate_and_penalize_edges(routes: typing.List[typing.List[int]], locations: numpy.ndarray, criterium: str, num_perturbations: int) -> typing.List[int]
:canonical: src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.evaluate_and_penalize_edges

```{autodoc2-docstring} src.policies.knowledge_guided_local_search.cost_evaluator.CostEvaluator.evaluate_and_penalize_edges
```

````

`````
