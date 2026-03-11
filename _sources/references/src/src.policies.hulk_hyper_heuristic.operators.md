# {py:mod}`src.policies.hulk_hyper_heuristic.operators`

```{py:module} src.policies.hulk_hyper_heuristic.operators
```

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKOperators <src.policies.hulk_hyper_heuristic.operators.HULKOperators>`
  - ```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators
    :summary:
    ```
````

### API

`````{py:class} HULKOperators(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.__init__
```

````{py:method} apply_unstring_type_i(solution: src.policies.hulk_hyper_heuristic.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hulk_hyper_heuristic.solution.Solution, typing.List[int]]
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_i

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_i
```

````

````{py:method} apply_unstring_type_ii(solution: src.policies.hulk_hyper_heuristic.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hulk_hyper_heuristic.solution.Solution, typing.List[int]]
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_ii

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_ii
```

````

````{py:method} apply_unstring_type_iii(solution: src.policies.hulk_hyper_heuristic.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hulk_hyper_heuristic.solution.Solution, typing.List[int]]
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_iii

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_iii
```

````

````{py:method} apply_unstring_type_iv(solution: src.policies.hulk_hyper_heuristic.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hulk_hyper_heuristic.solution.Solution, typing.List[int]]
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_iv

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_unstring_type_iv
```

````

````{py:method} _apply_unstring(solution: src.policies.hulk_hyper_heuristic.solution.Solution, n_remove: int, unstring_func) -> typing.Tuple[src.policies.hulk_hyper_heuristic.solution.Solution, typing.List[int]]
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators._apply_unstring

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators._apply_unstring
```

````

````{py:method} apply_string_repair(solution: src.policies.hulk_hyper_heuristic.solution.Solution, removed: typing.List[int], string_type: str) -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_string_repair

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_string_repair
```

````

````{py:method} _greedy_repair(solution: src.policies.hulk_hyper_heuristic.solution.Solution, removed: typing.List[int]) -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators._greedy_repair

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators._greedy_repair
```

````

````{py:method} _simple_2_opt(route: typing.List[int]) -> typing.Tuple[typing.List[int], bool]
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators._simple_2_opt

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators._simple_2_opt
```

````

````{py:method} _calc_route_distance(route: typing.List[int]) -> float
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators._calc_route_distance

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators._calc_route_distance
```

````

````{py:method} apply_2_opt(solution: src.policies.hulk_hyper_heuristic.solution.Solution) -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_2_opt

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_2_opt
```

````

````{py:method} apply_3_opt(solution: src.policies.hulk_hyper_heuristic.solution.Solution) -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_3_opt

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_3_opt
```

````

````{py:method} apply_swap(solution: src.policies.hulk_hyper_heuristic.solution.Solution) -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_swap

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_swap
```

````

````{py:method} apply_relocate(solution: src.policies.hulk_hyper_heuristic.solution.Solution) -> src.policies.hulk_hyper_heuristic.solution.Solution
:canonical: src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_relocate

```{autodoc2-docstring} src.policies.hulk_hyper_heuristic.operators.HULKOperators.apply_relocate
```

````

`````
