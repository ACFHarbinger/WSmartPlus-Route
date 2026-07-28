# {py:mod}`src.policies.helpers.operators.intra_route_local_search.k_opt`

```{py:module} src.policies.helpers.operators.intra_route_local_search.k_opt
```

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`two_opt_route <src.policies.helpers.operators.intra_route_local_search.k_opt.two_opt_route>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.two_opt_route
    :summary:
    ```
* - {py:obj}`three_opt_route <src.policies.helpers.operators.intra_route_local_search.k_opt.three_opt_route>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.three_opt_route
    :summary:
    ```
* - {py:obj}`move_2opt_intra <src.policies.helpers.operators.intra_route_local_search.k_opt.move_2opt_intra>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.move_2opt_intra
    :summary:
    ```
* - {py:obj}`move_3opt_intra <src.policies.helpers.operators.intra_route_local_search.k_opt.move_3opt_intra>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.move_3opt_intra
    :summary:
    ```
* - {py:obj}`move_kopt_intra <src.policies.helpers.operators.intra_route_local_search.k_opt.move_kopt_intra>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.move_kopt_intra
    :summary:
    ```
* - {py:obj}`_apply_2opt <src.policies.helpers.operators.intra_route_local_search.k_opt._apply_2opt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_2opt
    :summary:
    ```
* - {py:obj}`_apply_3opt <src.policies.helpers.operators.intra_route_local_search.k_opt._apply_3opt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_3opt
    :summary:
    ```
* - {py:obj}`_apply_kopt <src.policies.helpers.operators.intra_route_local_search.k_opt._apply_kopt>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_kopt
    :summary:
    ```
* - {py:obj}`_sample_cuts <src.policies.helpers.operators.intra_route_local_search.k_opt._sample_cuts>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._sample_cuts
    :summary:
    ```
* - {py:obj}`_get_segments <src.policies.helpers.operators.intra_route_local_search.k_opt._get_segments>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._get_segments
    :summary:
    ```
* - {py:obj}`_find_best_config <src.policies.helpers.operators.intra_route_local_search.k_opt._find_best_config>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._find_best_config
    :summary:
    ```
* - {py:obj}`_apply_config <src.policies.helpers.operators.intra_route_local_search.k_opt._apply_config>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_config
    :summary:
    ```
* - {py:obj}`_connection_cost <src.policies.helpers.operators.intra_route_local_search.k_opt._connection_cost>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._connection_cost
    :summary:
    ```
````

### API

````{py:function} two_opt_route(route: typing.List[int], dist_matrix: numpy.ndarray, max_iter: int = 200, exclude_depot: bool = False) -> typing.List[int]
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt.two_opt_route

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.two_opt_route
```
````

````{py:function} three_opt_route(route: typing.List[int], dist_matrix: numpy.ndarray, max_iter: int = 50, exclude_depot: bool = False) -> typing.List[int]
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt.three_opt_route

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.three_opt_route
```
````

````{py:function} move_2opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt.move_2opt_intra

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.move_2opt_intra
```
````

````{py:function} move_3opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt.move_3opt_intra

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.move_3opt_intra
```
````

````{py:function} move_kopt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, k: int = 2, rng: typing.Optional[random.Random] = None, exclude_depot: bool = False) -> bool
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt.move_kopt_intra

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt.move_kopt_intra
```
````

````{py:function} _apply_2opt(ls, u: int, v: int, r_u: int, p_u: int, p_v: int, exclude_depot: bool = False) -> bool
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._apply_2opt

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_2opt
```
````

````{py:function} _apply_3opt(ls, u: int, v: int, r_u: int, p_u: int, p_v: int, rng: random.Random, n_attempts: int = 5, exclude_depot: bool = False) -> bool
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._apply_3opt

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_3opt
```
````

````{py:function} _apply_kopt(ls, r_u: int, p_u: int, p_v: int, k: int, rng: random.Random, n_attempts: int = 5, exclude_depot: bool = False) -> bool
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._apply_kopt

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_kopt
```
````

````{py:function} _sample_cuts(n: int, p_u: int, p_v: int, k: int, rng: random.Random) -> typing.Optional[typing.List[int]]
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._sample_cuts

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._sample_cuts
```
````

````{py:function} _get_segments(route: typing.List[int], cuts: typing.List[int]) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._get_segments

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._get_segments
```
````

````{py:function} _find_best_config(ls, head: typing.List[int], middle: typing.List[typing.List[int]], tail: typing.List[int], original_cost: float, exclude_depot: bool = False) -> typing.Tuple[float, typing.Optional[typing.Tuple[typing.Tuple[int, ...], typing.Tuple[bool, ...]]]]
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._find_best_config

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._find_best_config
```
````

````{py:function} _apply_config(route: typing.List[int], head: typing.List[int], middle: typing.List[typing.List[int]], tail: typing.List[int], config: typing.Tuple[typing.Tuple[int, ...], typing.Tuple[bool, ...]]) -> None
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._apply_config

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._apply_config
```
````

````{py:function} _connection_cost(ls: typing.Any, head: typing.List[int], middle: typing.List[typing.List[int]], tail: typing.List[int], exclude_depot: bool = False) -> float
:canonical: src.policies.helpers.operators.intra_route_local_search.k_opt._connection_cost

```{autodoc2-docstring} src.policies.helpers.operators.intra_route_local_search.k_opt._connection_cost
```
````
