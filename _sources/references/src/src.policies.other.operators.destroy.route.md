# {py:mod}`src.policies.other.operators.destroy.route`

```{py:module} src.policies.other.operators.destroy.route
```

```{autodoc2-docstring} src.policies.other.operators.destroy.route
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`route_removal <src.policies.other.operators.destroy.route.route_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.route.route_removal
    :summary:
    ```
* - {py:obj}`_select_route <src.policies.other.operators.destroy.route._select_route>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.route._select_route
    :summary:
    ```
* - {py:obj}`_route_cost <src.policies.other.operators.destroy.route._route_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.route._route_cost
    :summary:
    ```
* - {py:obj}`route_profit_removal <src.policies.other.operators.destroy.route.route_profit_removal>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.route.route_profit_removal
    :summary:
    ```
* - {py:obj}`_select_profit_route <src.policies.other.operators.destroy.route._select_profit_route>`
  - ```{autodoc2-docstring} src.policies.other.operators.destroy.route._select_profit_route
    :summary:
    ```
````

### API

````{py:function} route_removal(routes: typing.List[typing.List[int]], strategy: str = 'random', dist_matrix: typing.Optional[numpy.ndarray] = None, wastes: typing.Optional[typing.Dict[int, float]] = None, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.route.route_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.route.route_removal
```
````

````{py:function} _select_route(routes: typing.List[typing.List[int]], strategy: str, dist_matrix: typing.Optional[numpy.ndarray], wastes: typing.Optional[typing.Dict[int, float]], rng: random.Random) -> int
:canonical: src.policies.other.operators.destroy.route._select_route

```{autodoc2-docstring} src.policies.other.operators.destroy.route._select_route
```
````

````{py:function} _route_cost(route: typing.List[int], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.other.operators.destroy.route._route_cost

```{autodoc2-docstring} src.policies.other.operators.destroy.route._route_cost
```
````

````{py:function} route_profit_removal(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, strategy: str = 'worst_profit', rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.destroy.route.route_profit_removal

```{autodoc2-docstring} src.policies.other.operators.destroy.route.route_profit_removal
```
````

````{py:function} _select_profit_route(routes: typing.List[typing.List[int]], strategy: str, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, rng: random.Random) -> int
:canonical: src.policies.other.operators.destroy.route._select_profit_route

```{autodoc2-docstring} src.policies.other.operators.destroy.route._select_profit_route
```
````
