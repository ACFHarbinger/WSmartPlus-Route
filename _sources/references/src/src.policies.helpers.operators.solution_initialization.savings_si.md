# {py:mod}`src.policies.helpers.operators.solution_initialization.savings_si`

```{py:module} src.policies.helpers.operators.solution_initialization.savings_si
```

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_compute_savings <src.policies.helpers.operators.solution_initialization.savings_si._compute_savings>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si._compute_savings
    :summary:
    ```
* - {py:obj}`_try_merge <src.policies.helpers.operators.solution_initialization.savings_si._try_merge>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si._try_merge
    :summary:
    ```
* - {py:obj}`build_savings_routes <src.policies.helpers.operators.solution_initialization.savings_si.build_savings_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si.build_savings_routes
    :summary:
    ```
````

### API

````{py:function} _compute_savings(eligible: typing.List[int], dist_matrix: numpy.ndarray) -> typing.List[typing.Tuple[float, int, int]]
:canonical: src.policies.helpers.operators.solution_initialization.savings_si._compute_savings

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si._compute_savings
```
````

````{py:function} _try_merge(s: float, i: int, j: int, route_of: typing.Dict[int, typing.List[int]], route_key: typing.Dict[int, int], load_of: typing.Dict[int, float], capacity: float) -> bool
:canonical: src.policies.helpers.operators.solution_initialization.savings_si._try_merge

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si._try_merge
```
````

````{py:function} build_savings_routes(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.solution_initialization.savings_si.build_savings_routes

```{autodoc2-docstring} src.policies.helpers.operators.solution_initialization.savings_si.build_savings_routes
```
````
