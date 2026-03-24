# {py:mod}`src.policies.other.operators.repair.geni`

```{py:module} src.policies.other.operators.repair.geni
```

```{autodoc2-docstring} src.policies.other.operators.repair.geni
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_rev_cost <src.policies.other.operators.repair.geni._get_rev_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni._get_rev_cost
    :summary:
    ```
* - {py:obj}`_apply_geni_move <src.policies.other.operators.repair.geni._apply_geni_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni._apply_geni_move
    :summary:
    ```
* - {py:obj}`_evaluate_route <src.policies.other.operators.repair.geni._evaluate_route>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni._evaluate_route
    :summary:
    ```
* - {py:obj}`_find_best_geni_move <src.policies.other.operators.repair.geni._find_best_geni_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni._find_best_geni_move
    :summary:
    ```
* - {py:obj}`geni_insertion <src.policies.other.operators.repair.geni.geni_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni.geni_insertion
    :summary:
    ```
* - {py:obj}`geni_profit_insertion <src.policies.other.operators.repair.geni.geni_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni.geni_profit_insertion
    :summary:
    ```
````

### API

````{py:function} _get_rev_cost(full_route: typing.List[int], start: int, end: int, dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.other.operators.repair.geni._get_rev_cost

```{autodoc2-docstring} src.policies.other.operators.repair.geni._get_rev_cost
```
````

````{py:function} _apply_geni_move(route: typing.List[int], u: int, i: int, j: int, m_type: str) -> typing.List[int]
:canonical: src.policies.other.operators.repair.geni._apply_geni_move

```{autodoc2-docstring} src.policies.other.operators.repair.geni._apply_geni_move
```
````

````{py:function} _evaluate_route(u: int, route: typing.List[int], dist_matrix: numpy.ndarray, neighborhood_size: int, revenue: typing.Optional[float], C: float, is_man: bool) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, str]]]
:canonical: src.policies.other.operators.repair.geni._evaluate_route

```{autodoc2-docstring} src.policies.other.operators.repair.geni._evaluate_route
```
````

````{py:function} _find_best_geni_move(u: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, u_waste: float, capacity: float, neighborhood_size: int, revenue: typing.Optional[float] = None, C: float = 1.0, mandatory_set: typing.Optional[typing.Set[int]] = None) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int, str]]]
:canonical: src.policies.other.operators.repair.geni._find_best_geni_move

```{autodoc2-docstring} src.policies.other.operators.repair.geni._find_best_geni_move
```
````

````{py:function} geni_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, neighborhood_size: int = 5, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.geni.geni_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.geni.geni_insertion
```
````

````{py:function} geni_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, neighborhood_size: int = 5, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False, rng: typing.Optional[random.Random] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.geni.geni_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.geni.geni_profit_insertion
```
````
