# {py:mod}`src.policies.helpers.operators.repair.geni`

```{py:module} src.policies.helpers.operators.repair.geni
```

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_rev_cost <src.policies.helpers.operators.repair.geni._get_rev_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._get_rev_cost
    :summary:
    ```
* - {py:obj}`_apply_geni_move <src.policies.helpers.operators.repair.geni._apply_geni_move>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._apply_geni_move
    :summary:
    ```
* - {py:obj}`_evaluate_route <src.policies.helpers.operators.repair.geni._evaluate_route>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._evaluate_route
    :summary:
    ```
* - {py:obj}`_find_best_geni_move <src.policies.helpers.operators.repair.geni._find_best_geni_move>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._find_best_geni_move
    :summary:
    ```
* - {py:obj}`geni_insertion <src.policies.helpers.operators.repair.geni.geni_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.geni.geni_insertion
    :summary:
    ```
* - {py:obj}`geni_profit_insertion <src.policies.helpers.operators.repair.geni.geni_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.repair.geni.geni_profit_insertion
    :summary:
    ```
````

### API

````{py:function} _get_rev_cost(forward_cost: numpy.ndarray, backward_cost: numpy.ndarray, start: int, end: int) -> float
:canonical: src.policies.helpers.operators.repair.geni._get_rev_cost

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._get_rev_cost
```
````

````{py:function} _apply_geni_move(route: typing.List[int], u: int, i: int, j: int, k: int, l: int, m_type: str) -> typing.List[int]
:canonical: src.policies.helpers.operators.repair.geni._apply_geni_move

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._apply_geni_move
```
````

````{py:function} _evaluate_route(u: int, route: typing.List[int], dist_matrix: numpy.ndarray, neighborhood_size: int, revenue: typing.Optional[float], C: float, is_man: bool, use_deterministic_p_neighborhood: bool = False) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int, int, str]]]
:canonical: src.policies.helpers.operators.repair.geni._evaluate_route

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._evaluate_route
```
````

````{py:function} _find_best_geni_move(u: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, u_waste: float, capacity: float, neighborhood_size: int, revenue: typing.Optional[float] = None, C: float = 1.0, mandatory_set: typing.Optional[typing.Set[int]] = None, use_deterministic_p_neighborhood: bool = False) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, int, int, int, int, str, bool]]]
:canonical: src.policies.helpers.operators.repair.geni._find_best_geni_move

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni._find_best_geni_move
```
````

````{py:function} geni_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, neighborhood_size: int = 5, mandatory_nodes: typing.Optional[typing.List[int]] = None, rng: typing.Optional[random.Random] = None, expand_pool: bool = False, use_deterministic_p_neighborhood: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.repair.geni.geni_insertion

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni.geni_insertion
```
````

````{py:function} geni_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, neighborhood_size: int = 5, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False, rng: typing.Optional[random.Random] = None, use_deterministic_p_neighborhood: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.repair.geni.geni_profit_insertion

```{autodoc2-docstring} src.policies.helpers.operators.repair.geni.geni_profit_insertion
```
````
