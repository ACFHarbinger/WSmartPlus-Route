# {py:mod}`src.policies.other.operators.intra_route.geni`

```{py:module} src.policies.other.operators.intra_route.geni
```

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`geni_insert <src.policies.other.operators.intra_route.geni.geni_insert>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.geni.geni_insert
    :summary:
    ```
* - {py:obj}`_get_nearest_in_route <src.policies.other.operators.intra_route.geni._get_nearest_in_route>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._get_nearest_in_route
    :summary:
    ```
* - {py:obj}`_evaluate_type_i <src.policies.other.operators.intra_route.geni._evaluate_type_i>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._evaluate_type_i
    :summary:
    ```
* - {py:obj}`_evaluate_type_ii <src.policies.other.operators.intra_route.geni._evaluate_type_ii>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._evaluate_type_ii
    :summary:
    ```
* - {py:obj}`_apply_type_i <src.policies.other.operators.intra_route.geni._apply_type_i>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._apply_type_i
    :summary:
    ```
* - {py:obj}`_apply_type_ii <src.policies.other.operators.intra_route.geni._apply_type_ii>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._apply_type_ii
    :summary:
    ```
````

### API

````{py:function} geni_insert(ls: typing.Any, node: int, r_idx: int, neighborhood_size: int = 5) -> bool
:canonical: src.policies.other.operators.intra_route.geni.geni_insert

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni.geni_insert
```
````

````{py:function} _get_nearest_in_route(d, node: int, route: typing.List[int], k: int) -> typing.List[int]
:canonical: src.policies.other.operators.intra_route.geni._get_nearest_in_route

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._get_nearest_in_route
```
````

````{py:function} _evaluate_type_i(ls: typing.Any, route: typing.List[int], node: int, pi: int, pj: int) -> float
:canonical: src.policies.other.operators.intra_route.geni._evaluate_type_i

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._evaluate_type_i
```
````

````{py:function} _evaluate_type_ii(ls: typing.Any, route: typing.List[int], node: int, pi: int, pj: int) -> float
:canonical: src.policies.other.operators.intra_route.geni._evaluate_type_ii

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._evaluate_type_ii
```
````

````{py:function} _apply_type_i(ls: typing.Any, route: typing.List[int], node: int, pi: int, pj: int, r_idx: int) -> None
:canonical: src.policies.other.operators.intra_route.geni._apply_type_i

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._apply_type_i
```
````

````{py:function} _apply_type_ii(ls: typing.Any, route: typing.List[int], node: int, pi: int, pj: int, r_idx: int) -> None
:canonical: src.policies.other.operators.intra_route.geni._apply_type_ii

```{autodoc2-docstring} src.policies.other.operators.intra_route.geni._apply_type_ii
```
````
