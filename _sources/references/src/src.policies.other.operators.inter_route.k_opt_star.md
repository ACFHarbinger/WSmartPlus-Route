# {py:mod}`src.policies.other.operators.inter_route.k_opt_star`

```{py:module} src.policies.other.operators.inter_route.k_opt_star
```

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`move_2opt_star <src.policies.other.operators.inter_route.k_opt_star.move_2opt_star>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.move_2opt_star
    :summary:
    ```
* - {py:obj}`move_3opt_star <src.policies.other.operators.inter_route.k_opt_star.move_3opt_star>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.move_3opt_star
    :summary:
    ```
* - {py:obj}`move_kopt_star <src.policies.other.operators.inter_route.k_opt_star.move_kopt_star>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.move_kopt_star
    :summary:
    ```
* - {py:obj}`_extract_route_parts <src.policies.other.operators.inter_route.k_opt_star._extract_route_parts>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star._extract_route_parts
    :summary:
    ```
* - {py:obj}`_find_best_tail_permutation <src.policies.other.operators.inter_route.k_opt_star._find_best_tail_permutation>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star._find_best_tail_permutation
    :summary:
    ```
* - {py:obj}`_apply_tail_permutation <src.policies.other.operators.inter_route.k_opt_star._apply_tail_permutation>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star._apply_tail_permutation
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`CutPoint <src.policies.other.operators.inter_route.k_opt_star.CutPoint>`
  - ```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.CutPoint
    :summary:
    ```
````

### API

````{py:data} CutPoint
:canonical: src.policies.other.operators.inter_route.k_opt_star.CutPoint
:value: >
   None

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.CutPoint
```

````

````{py:function} move_2opt_star(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.other.operators.inter_route.k_opt_star.move_2opt_star

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.move_2opt_star
```
````

````{py:function} move_3opt_star(ls, u: int, v: int, w: int, r_u: int, p_u: int, r_v: int, p_v: int, r_w: int, p_w: int) -> bool
:canonical: src.policies.other.operators.inter_route.k_opt_star.move_3opt_star

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.move_3opt_star
```
````

````{py:function} move_kopt_star(ls, cuts: typing.List[src.policies.other.operators.inter_route.k_opt_star.CutPoint]) -> bool
:canonical: src.policies.other.operators.inter_route.k_opt_star.move_kopt_star

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star.move_kopt_star
```
````

````{py:function} _extract_route_parts(ls, cuts: typing.List[src.policies.other.operators.inter_route.k_opt_star.CutPoint]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[typing.List[int]], typing.List[float], typing.List[float], float]
:canonical: src.policies.other.operators.inter_route.k_opt_star._extract_route_parts

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star._extract_route_parts
```
````

````{py:function} _find_best_tail_permutation(ls, cuts: typing.List[src.policies.other.operators.inter_route.k_opt_star.CutPoint], heads: typing.List[typing.List[int]], tails: typing.List[typing.List[int]], head_loads: typing.List[float], tail_loads: typing.List[float], original_cost: float) -> typing.Tuple[float, typing.Optional[typing.Tuple[int, ...]]]
:canonical: src.policies.other.operators.inter_route.k_opt_star._find_best_tail_permutation

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star._find_best_tail_permutation
```
````

````{py:function} _apply_tail_permutation(ls, cuts: typing.List[src.policies.other.operators.inter_route.k_opt_star.CutPoint], heads: typing.List[typing.List[int]], tails: typing.List[typing.List[int]], perm: typing.Tuple[int, ...]) -> None
:canonical: src.policies.other.operators.inter_route.k_opt_star._apply_tail_permutation

```{autodoc2-docstring} src.policies.other.operators.inter_route.k_opt_star._apply_tail_permutation
```
````
