# {py:mod}`src.policies.helpers.operators.intensification.or_opt_steepest`

```{py:module} src.policies.helpers.operators.intensification.or_opt_steepest
```

```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_find_best_or_opt <src.policies.helpers.operators.intensification.or_opt_steepest._find_best_or_opt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest._find_best_or_opt
    :summary:
    ```
* - {py:obj}`or_opt_steepest <src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest
    :summary:
    ```
* - {py:obj}`or_opt_steepest_profit <src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest_profit
    :summary:
    ```
````

### API

````{py:function} _find_best_or_opt(routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, C: float, chain_lengths: typing.Tuple[int, ...]) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int, int, int]]]
:canonical: src.policies.helpers.operators.intensification.or_opt_steepest._find_best_or_opt

```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest._find_best_or_opt
```
````

````{py:function} or_opt_steepest(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, C: float = 1.0, chain_lengths: typing.Tuple[int, ...] = (1, 2, 3), max_iter: int = 500) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest

```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest
```
````

````{py:function} or_opt_steepest_profit(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, chain_lengths: typing.Tuple[int, ...] = (1, 2, 3), max_iter: int = 500) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest_profit

```{autodoc2-docstring} src.policies.helpers.operators.intensification.or_opt_steepest.or_opt_steepest_profit
```
````
