# {py:mod}`src.policies.exact_stochastic_dynamic_programming.cop_evaluator`

```{py:module} src.policies.exact_stochastic_dynamic_programming.cop_evaluator
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`COPEvaluator <src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator>`
  - ```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator
    :summary:
    ```
````

### API

`````{py:class} COPEvaluator(dist_matrix: numpy.ndarray, num_nodes: int)
:canonical: src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator.__init__
```

````{py:method} _precompute_all_subsets()
:canonical: src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator._precompute_all_subsets

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator._precompute_all_subsets
```

````

````{py:method} get_route_cost(subset: typing.FrozenSet[int]) -> float
:canonical: src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator.get_route_cost

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator.get_route_cost
```

````

````{py:method} get_feasible_actions(discrete_state: typing.Tuple[int, ...], capacity: float, L: int) -> typing.List[typing.FrozenSet[int]]
:canonical: src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator.get_feasible_actions

```{autodoc2-docstring} src.policies.exact_stochastic_dynamic_programming.cop_evaluator.COPEvaluator.get_feasible_actions
```

````

`````
