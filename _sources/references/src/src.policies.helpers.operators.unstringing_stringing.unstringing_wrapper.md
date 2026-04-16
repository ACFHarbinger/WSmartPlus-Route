# {py:mod}`src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper`

```{py:module} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper
```

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_evaluate_routes <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._evaluate_routes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._evaluate_routes
    :summary:
    ```
* - {py:obj}`_apply_unstring_op <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._apply_unstring_op>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._apply_unstring_op
    :summary:
    ```
* - {py:obj}`_try_unstring_removal <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal
    :summary:
    ```
* - {py:obj}`_try_unstring_removal_deterministic <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal_deterministic>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal_deterministic
    :summary:
    ```
* - {py:obj}`unstringing_removal_wrapper <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal_wrapper>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal_wrapper
    :summary:
    ```
* - {py:obj}`unstringing_removal <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal
    :summary:
    ```
* - {py:obj}`unstringing_profit_removal <src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_profit_removal
    :summary:
    ```
````

### API

````{py:function} _evaluate_routes(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._evaluate_routes

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._evaluate_routes
```
````

````{py:function} _apply_unstring_op(route: typing.List[int], unstring_type: int, params: typing.Tuple, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, profit_mode: bool) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._apply_unstring_op

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._apply_unstring_op
```
````

````{py:function} _try_unstring_removal(routes: typing.List[typing.List[int]], r_idx: int, n_idx: int, unstring_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, rng: random.Random, profit_mode: bool = False) -> typing.Optional[typing.Tuple[typing.List[typing.List[int]], float]]
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal
```
````

````{py:function} _try_unstring_removal_deterministic(routes: typing.List[typing.List[int]], r_idx: int, n_idx: int, unstring_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, neighborhood_size: int, profit_mode: bool = False) -> typing.Optional[typing.Tuple[typing.List[typing.List[int]], float]]
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal_deterministic

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper._try_unstring_removal_deterministic
```
````

````{py:function} unstringing_removal_wrapper(routes: typing.List[typing.List[int]], n_remove: int, unstring_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 0.0, C: float = 1.0, rng: typing.Optional[random.Random] = None, profit_mode: bool = False, target_node: typing.Optional[int] = None, use_alns_fallback: bool = False, random_us_sampling: bool = True, neighborhood_size: int = 5) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal_wrapper

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal_wrapper
```
````

````{py:function} unstringing_removal(routes: typing.List[typing.List[int]], n_remove: int, unstring_type: int, dist_matrix: numpy.ndarray, rng: typing.Optional[random.Random] = None, target_node: typing.Optional[int] = None, use_alns_fallback: bool = False, random_us_sampling: bool = True, neighborhood_size: int = 5) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_removal
```
````

````{py:function} unstringing_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, unstring_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None, target_node: typing.Optional[int] = None, use_alns_fallback: bool = False, random_us_sampling: bool = True, neighborhood_size: int = 5) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.unstringing_stringing.unstringing_wrapper.unstringing_profit_removal
```
````
