# {py:mod}`src.policies.other.operators.unstringing_stringing.stringing_wrapper`

```{py:module} src.policies.other.operators.unstringing_stringing.stringing_wrapper
```

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_evaluate_routes <src.policies.other.operators.unstringing_stringing.stringing_wrapper._evaluate_routes>`
  - ```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper._evaluate_routes
    :summary:
    ```
* - {py:obj}`_apply_stringing_op <src.policies.other.operators.unstringing_stringing.stringing_wrapper._apply_stringing_op>`
  - ```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper._apply_stringing_op
    :summary:
    ```
* - {py:obj}`_try_string_insertion <src.policies.other.operators.unstringing_stringing.stringing_wrapper._try_string_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper._try_string_insertion
    :summary:
    ```
* - {py:obj}`stringing_insertion_wrapper <src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion_wrapper>`
  - ```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion_wrapper
    :summary:
    ```
* - {py:obj}`stringing_insertion <src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion
    :summary:
    ```
* - {py:obj}`stringing_profit_insertion <src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_profit_insertion
    :summary:
    ```
````

### API

````{py:function} _evaluate_routes(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.other.operators.unstringing_stringing.stringing_wrapper._evaluate_routes

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper._evaluate_routes
```
````

````{py:function} _apply_stringing_op(route: typing.List[int], node: int, string_type: int, params: typing.Tuple, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, profit_mode: bool) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.other.operators.unstringing_stringing.stringing_wrapper._apply_stringing_op

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper._apply_stringing_op
```
````

````{py:function} _try_string_insertion(routes: typing.List[typing.List[int]], node: int, r_idx: int, string_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, rng: random.Random, profit_mode: bool = False) -> typing.Optional[typing.Tuple[typing.List[typing.List[int]], float]]
:canonical: src.policies.other.operators.unstringing_stringing.stringing_wrapper._try_string_insertion

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper._try_string_insertion
```
````

````{py:function} stringing_insertion_wrapper(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], string_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float = 0.0, C: float = 1.0, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, profit_mode: bool = False, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion_wrapper

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion_wrapper
```
````

````{py:function} stringing_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], string_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_insertion
```
````

````{py:function} stringing_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], string_type: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.unstringing_stringing.stringing_wrapper.stringing_profit_insertion
```
````
