# {py:mod}`src.policies.hyper_heuristic_us_lk.operators`

```{py:module} src.policies.hyper_heuristic_us_lk.operators
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HULKOperators <src.policies.hyper_heuristic_us_lk.operators.HULKOperators>`
  - ```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators
    :summary:
    ```
````

### API

`````{py:class} HULKOperators(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None, expand_pool: bool = False, profit_aware_operators: bool = False)
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.__init__
```

````{py:method} apply_unstring_type_i(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_i

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_i
```

````

````{py:method} apply_unstring_type_ii(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_ii

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_ii
```

````

````{py:method} apply_unstring_type_iii(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_iii

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_iii
```

````

````{py:method} apply_unstring_type_iv(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_iv

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_type_iv
```

````

````{py:method} apply_unstring_shaw(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_shaw

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_shaw
```

````

````{py:method} apply_unstring_string(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_string

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_unstring_string
```

````

````{py:method} _apply_unstring_wrapper(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, n_remove: int, unstring_type: int) -> typing.Tuple[src.policies.hyper_heuristic_us_lk.solution.Solution, typing.List[int]]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators._apply_unstring_wrapper

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators._apply_unstring_wrapper
```

````

````{py:method} apply_string_repair(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, removed: typing.List[int], string_type: str) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_string_repair

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_string_repair
```

````

````{py:method} _greedy_repair(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, removed: typing.List[int]) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators._greedy_repair

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators._greedy_repair
```

````

````{py:method} _regret_2_repair(solution: src.policies.hyper_heuristic_us_lk.solution.Solution, removed: typing.List[int]) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators._regret_2_repair

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators._regret_2_repair
```

````

````{py:method} _simple_2_opt(route: typing.List[int]) -> typing.Tuple[typing.List[int], bool]
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators._simple_2_opt

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators._simple_2_opt
```

````

````{py:method} _calc_route_distance(route: typing.List[int]) -> float
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators._calc_route_distance

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators._calc_route_distance
```

````

````{py:method} apply_2_opt(solution: src.policies.hyper_heuristic_us_lk.solution.Solution) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_2_opt

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_2_opt
```

````

````{py:method} apply_3_opt(solution: src.policies.hyper_heuristic_us_lk.solution.Solution) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_3_opt

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_3_opt
```

````

````{py:method} apply_swap(solution: src.policies.hyper_heuristic_us_lk.solution.Solution) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_swap

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_swap
```

````

````{py:method} apply_relocate(solution: src.policies.hyper_heuristic_us_lk.solution.Solution) -> src.policies.hyper_heuristic_us_lk.solution.Solution
:canonical: src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_relocate

```{autodoc2-docstring} src.policies.hyper_heuristic_us_lk.operators.HULKOperators.apply_relocate
```

````

`````
