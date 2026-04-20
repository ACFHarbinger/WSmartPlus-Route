# {py:mod}`src.policies.helpers.operators.improvement_descent.node_exchange_steepest`

```{py:module} src.policies.helpers.operators.improvement_descent.node_exchange_steepest
```

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_find_best_swap <src.policies.helpers.operators.improvement_descent.node_exchange_steepest._find_best_swap>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest._find_best_swap
    :summary:
    ```
* - {py:obj}`node_exchange_steepest <src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest
    :summary:
    ```
* - {py:obj}`node_exchange_steepest_profit <src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest_profit
    :summary:
    ```
````

### API

````{py:function} _find_best_swap(routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, C: float) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int, int]]]
:canonical: src.policies.helpers.operators.improvement_descent.node_exchange_steepest._find_best_swap

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest._find_best_swap
```
````

````{py:function} node_exchange_steepest(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, C: float = 1.0, max_iter: int = 300) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest
```
````

````{py:function} node_exchange_steepest_profit(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, max_iter: int = 300) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest_profit

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.node_exchange_steepest.node_exchange_steepest_profit
```
````
