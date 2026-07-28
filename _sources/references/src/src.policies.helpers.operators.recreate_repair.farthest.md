# {py:mod}`src.policies.helpers.operators.recreate_repair.farthest`

```{py:module} src.policies.helpers.operators.recreate_repair.farthest
```

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_farthest_node <src.policies.helpers.operators.recreate_repair.farthest._get_farthest_node>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest._get_farthest_node
    :summary:
    ```
* - {py:obj}`_find_cheapest_insertion <src.policies.helpers.operators.recreate_repair.farthest._find_cheapest_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest._find_cheapest_insertion
    :summary:
    ```
* - {py:obj}`_find_nearest_insertion <src.policies.helpers.operators.recreate_repair.farthest._find_nearest_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest._find_nearest_insertion
    :summary:
    ```
* - {py:obj}`farthest_insertion <src.policies.helpers.operators.recreate_repair.farthest.farthest_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest.farthest_insertion
    :summary:
    ```
* - {py:obj}`farthest_profit_insertion <src.policies.helpers.operators.recreate_repair.farthest.farthest_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest.farthest_profit_insertion
    :summary:
    ```
````

### API

````{py:function} _get_farthest_node(unassigned: typing.List[int], routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> typing.Optional[int]
:canonical: src.policies.helpers.operators.recreate_repair.farthest._get_farthest_node

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest._get_farthest_node
```
````

````{py:function} _find_cheapest_insertion(farthest_node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, capacity: float, node_waste: float, revenue: float, is_mandatory: bool, R: typing.Optional[float] = None, C: typing.Optional[float] = None) -> typing.Tuple[int, int, float]
:canonical: src.policies.helpers.operators.recreate_repair.farthest._find_cheapest_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest._find_cheapest_insertion
```
````

````{py:function} _find_nearest_insertion(farthest_node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, capacity: float, node_waste: float) -> typing.Tuple[int, int, float]
:canonical: src.policies.helpers.operators.recreate_repair.farthest._find_nearest_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest._find_nearest_insertion
```
````

````{py:function} farthest_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.recreate_repair.farthest.farthest_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest.farthest_insertion
```
````

````{py:function} farthest_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.recreate_repair.farthest.farthest_profit_insertion

```{autodoc2-docstring} src.policies.helpers.operators.recreate_repair.farthest.farthest_profit_insertion
```
````
