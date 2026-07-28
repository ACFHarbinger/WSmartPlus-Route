# {py:mod}`src.policies.helpers.operators.improvement_descent.steepest_two_opt`

```{py:module} src.policies.helpers.operators.improvement_descent.steepest_two_opt
```

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_find_best_2opt <src.policies.helpers.operators.improvement_descent.steepest_two_opt._find_best_2opt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt._find_best_2opt
    :summary:
    ```
* - {py:obj}`two_opt_steepest <src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest
    :summary:
    ```
* - {py:obj}`two_opt_steepest_profit <src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest_profit
    :summary:
    ```
````

### API

````{py:function} _find_best_2opt(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, C: float) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int]]]
:canonical: src.policies.helpers.operators.improvement_descent.steepest_two_opt._find_best_2opt

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt._find_best_2opt
```
````

````{py:function} two_opt_steepest(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, C: float = 1.0, max_iter: int = 500) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest
```
````

````{py:function} two_opt_steepest_profit(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, max_iter: int = 500) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest_profit

```{autodoc2-docstring} src.policies.helpers.operators.improvement_descent.steepest_two_opt.two_opt_steepest_profit
```
````
