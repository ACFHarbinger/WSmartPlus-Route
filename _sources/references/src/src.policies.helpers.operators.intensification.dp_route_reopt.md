# {py:mod}`src.policies.helpers.operators.intensification.dp_route_reopt`

```{py:module} src.policies.helpers.operators.intensification.dp_route_reopt
```

```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_held_karp <src.policies.helpers.operators.intensification.dp_route_reopt._held_karp>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt._held_karp
    :summary:
    ```
* - {py:obj}`dp_route_reopt <src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt
    :summary:
    ```
* - {py:obj}`dp_route_reopt_profit <src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt_profit>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt_profit
    :summary:
    ```
````

### API

````{py:function} _held_karp(nodes: typing.List[int], dist_matrix: numpy.ndarray) -> typing.Tuple[float, typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.dp_route_reopt._held_karp

```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt._held_karp
```
````

````{py:function} dp_route_reopt(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, max_nodes: int = 20, max_iter: int = 1) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt

```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt
```
````

````{py:function} dp_route_reopt_profit(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, max_nodes: int = 20, max_iter: int = 1) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt_profit

```{autodoc2-docstring} src.policies.helpers.operators.intensification.dp_route_reopt.dp_route_reopt_profit
```
````
