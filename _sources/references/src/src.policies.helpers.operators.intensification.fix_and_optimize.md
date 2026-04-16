# {py:mod}`src.policies.helpers.operators.intensification.fix_and_optimize`

```{py:module} src.policies.helpers.operators.intensification.fix_and_optimize
```

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_route_distance <src.policies.helpers.operators.intensification.fix_and_optimize._route_distance>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._route_distance
    :summary:
    ```
* - {py:obj}`_route_profit <src.policies.helpers.operators.intensification.fix_and_optimize._route_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._route_profit
    :summary:
    ```
* - {py:obj}`_solve_cvrp_mip <src.policies.helpers.operators.intensification.fix_and_optimize._solve_cvrp_mip>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._solve_cvrp_mip
    :summary:
    ```
* - {py:obj}`_solve_vrpp_mip <src.policies.helpers.operators.intensification.fix_and_optimize._solve_vrpp_mip>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._solve_vrpp_mip
    :summary:
    ```
* - {py:obj}`fix_and_optimize <src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize
    :summary:
    ```
* - {py:obj}`fix_and_optimize_profit <src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize_profit
    :summary:
    ```
````

### API

````{py:function} _route_distance(route: typing.List[int], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.helpers.operators.intensification.fix_and_optimize._route_distance

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._route_distance
```
````

````{py:function} _route_profit(route: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float) -> float
:canonical: src.policies.helpers.operators.intensification.fix_and_optimize._route_profit

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._route_profit
```
````

````{py:function} _solve_cvrp_mip(free_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, n_vehicles: int, time_limit: float, seed: int) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.fix_and_optimize._solve_cvrp_mip

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._solve_cvrp_mip
```
````

````{py:function} _solve_vrpp_mip(free_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, n_vehicles: int, time_limit: float, seed: int, mandatory_nodes: typing.Optional[typing.Set[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.fix_and_optimize._solve_vrpp_mip

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize._solve_vrpp_mip
```
````

````{py:function} fix_and_optimize(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, n_free: typing.Optional[int] = None, free_fraction: float = 0.3, time_limit: float = 30.0, seed: int = 42) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize
```
````

````{py:function} fix_and_optimize_profit(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, n_free: typing.Optional[int] = None, free_fraction: float = 0.3, time_limit: float = 30.0, seed: int = 42, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize_profit

```{autodoc2-docstring} src.policies.helpers.operators.intensification.fix_and_optimize.fix_and_optimize_profit
```
````
