# {py:mod}`src.policies.other.operators.repair.nearest`

```{py:module} src.policies.other.operators.repair.nearest
```

```{autodoc2-docstring} src.policies.other.operators.repair.nearest
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_nearest_node <src.policies.other.operators.repair.nearest._get_nearest_node>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.nearest._get_nearest_node
    :summary:
    ```
* - {py:obj}`_find_cheapest_insertion <src.policies.other.operators.repair.nearest._find_cheapest_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.nearest._find_cheapest_insertion
    :summary:
    ```
* - {py:obj}`nearest_insertion <src.policies.other.operators.repair.nearest.nearest_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.nearest.nearest_insertion
    :summary:
    ```
* - {py:obj}`_find_cheapest_insertion_dist <src.policies.other.operators.repair.nearest._find_cheapest_insertion_dist>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.nearest._find_cheapest_insertion_dist
    :summary:
    ```
* - {py:obj}`nearest_profit_insertion <src.policies.other.operators.repair.nearest.nearest_profit_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.nearest.nearest_profit_insertion
    :summary:
    ```
````

### API

````{py:function} _get_nearest_node(unassigned: typing.List[int], routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> typing.Optional[int]
:canonical: src.policies.other.operators.repair.nearest._get_nearest_node

```{autodoc2-docstring} src.policies.other.operators.repair.nearest._get_nearest_node
```
````

````{py:function} _find_cheapest_insertion(node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, capacity: float, node_waste: float, revenue: float, is_mandatory: bool, R: typing.Optional[float] = None, C: float = 1.0) -> typing.Tuple[int, int, float]
:canonical: src.policies.other.operators.repair.nearest._find_cheapest_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.nearest._find_cheapest_insertion
```
````

````{py:function} nearest_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = True) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.nearest.nearest_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.nearest.nearest_insertion
```
````

````{py:function} _find_cheapest_insertion_dist(node: int, routes: typing.List[typing.List[int]], loads: typing.List[float], dist_matrix: numpy.ndarray, capacity: float, node_waste: float) -> typing.Tuple[int, int, float]
:canonical: src.policies.other.operators.repair.nearest._find_cheapest_insertion_dist

```{autodoc2-docstring} src.policies.other.operators.repair.nearest._find_cheapest_insertion_dist
```
````

````{py:function} nearest_profit_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.nearest.nearest_profit_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.nearest.nearest_profit_insertion
```
````
