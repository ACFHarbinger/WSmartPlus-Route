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

* - {py:obj}`geni_insertion <src.policies.other.operators.repair.geni.geni_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni.geni_insertion
    :summary:
    ```
* - {py:obj}`_evaluate_route_insertion <src.policies.other.operators.repair.geni._evaluate_route_insertion>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni._evaluate_route_insertion
    :summary:
    ```
* - {py:obj}`_apply_geni_move <src.policies.other.operators.repair.geni._apply_geni_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.repair.geni._apply_geni_move
    :summary:
    ```
````

### API

````{py:function} geni_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: dict, capacity: float, R: float, C: float, neighborhood_size: int = 5) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.repair.geni.geni_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.geni.geni_insertion
```
````

````{py:function} _evaluate_route_insertion(node: int, r_idx: int, route: typing.List[int], dist_matrix: numpy.ndarray, wastes: dict, capacity: float, revenue: float, node_waste: float, C: float, neighborhood_size: int) -> typing.Tuple[float, typing.Any]
:canonical: src.policies.other.operators.repair.geni._evaluate_route_insertion

```{autodoc2-docstring} src.policies.other.operators.repair.geni._evaluate_route_insertion
```
````

````{py:function} _apply_geni_move(routes: typing.List[typing.List[int]], u: int, r_idx: int, i: int, j: int, m_type: str)
:canonical: src.policies.other.operators.repair.geni._apply_geni_move

```{autodoc2-docstring} src.policies.other.operators.repair.geni._apply_geni_move
```
````
