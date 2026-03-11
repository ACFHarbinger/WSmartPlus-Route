# {py:mod}`src.policies.other.operators.intra_route.k_opt`

```{py:module} src.policies.other.operators.intra_route.k_opt
```

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`move_2opt_intra <src.policies.other.operators.intra_route.k_opt.move_2opt_intra>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt.move_2opt_intra
    :summary:
    ```
* - {py:obj}`move_3opt_intra <src.policies.other.operators.intra_route.k_opt.move_3opt_intra>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt.move_3opt_intra
    :summary:
    ```
* - {py:obj}`move_kopt_intra <src.policies.other.operators.intra_route.k_opt.move_kopt_intra>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt.move_kopt_intra
    :summary:
    ```
* - {py:obj}`_apply_2opt <src.policies.other.operators.intra_route.k_opt._apply_2opt>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_2opt
    :summary:
    ```
* - {py:obj}`_apply_3opt <src.policies.other.operators.intra_route.k_opt._apply_3opt>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_3opt
    :summary:
    ```
* - {py:obj}`_apply_kopt <src.policies.other.operators.intra_route.k_opt._apply_kopt>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_kopt
    :summary:
    ```
* - {py:obj}`_sample_cuts <src.policies.other.operators.intra_route.k_opt._sample_cuts>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._sample_cuts
    :summary:
    ```
* - {py:obj}`_get_segments <src.policies.other.operators.intra_route.k_opt._get_segments>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._get_segments
    :summary:
    ```
* - {py:obj}`_find_best_config <src.policies.other.operators.intra_route.k_opt._find_best_config>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._find_best_config
    :summary:
    ```
* - {py:obj}`_apply_config <src.policies.other.operators.intra_route.k_opt._apply_config>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_config
    :summary:
    ```
* - {py:obj}`_connection_cost <src.policies.other.operators.intra_route.k_opt._connection_cost>`
  - ```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._connection_cost
    :summary:
    ```
````

### API

````{py:function} move_2opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.other.operators.intra_route.k_opt.move_2opt_intra

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt.move_2opt_intra
```
````

````{py:function} move_3opt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool
:canonical: src.policies.other.operators.intra_route.k_opt.move_3opt_intra

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt.move_3opt_intra
```
````

````{py:function} move_kopt_intra(ls, u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, k: int = 2, rng: typing.Optional[random.Random] = None) -> bool
:canonical: src.policies.other.operators.intra_route.k_opt.move_kopt_intra

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt.move_kopt_intra
```
````

````{py:function} _apply_2opt(ls, u: int, v: int, r_u: int, p_u: int, p_v: int) -> bool
:canonical: src.policies.other.operators.intra_route.k_opt._apply_2opt

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_2opt
```
````

````{py:function} _apply_3opt(ls, u: int, v: int, r_u: int, p_u: int, p_v: int, rng: random.Random, n_attempts=5) -> bool
:canonical: src.policies.other.operators.intra_route.k_opt._apply_3opt

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_3opt
```
````

````{py:function} _apply_kopt(ls, r_u: int, p_u: int, p_v: int, k: int, rng: random.Random, n_attempts: int = 5) -> bool
:canonical: src.policies.other.operators.intra_route.k_opt._apply_kopt

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_kopt
```
````

````{py:function} _sample_cuts(n: int, p_u: int, p_v: int, k: int, rng: random.Random) -> typing.Optional[typing.List[int]]
:canonical: src.policies.other.operators.intra_route.k_opt._sample_cuts

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._sample_cuts
```
````

````{py:function} _get_segments(route: typing.List[int], cuts: typing.List[int]) -> typing.Tuple[typing.List[int], typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.other.operators.intra_route.k_opt._get_segments

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._get_segments
```
````

````{py:function} _find_best_config(ls, head: typing.List[int], middle: typing.List[typing.List[int]], tail: typing.List[int], original_cost: float) -> typing.Tuple[float, typing.Optional[typing.Tuple[typing.Tuple[int, ...], typing.Tuple[bool, ...]]]]
:canonical: src.policies.other.operators.intra_route.k_opt._find_best_config

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._find_best_config
```
````

````{py:function} _apply_config(route: typing.List[int], head: typing.List[int], middle: typing.List[typing.List[int]], tail: typing.List[int], config: typing.Tuple[typing.Tuple[int, ...], typing.Tuple[bool, ...]]) -> None
:canonical: src.policies.other.operators.intra_route.k_opt._apply_config

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._apply_config
```
````

````{py:function} _connection_cost(d, head: typing.List[int], middle: typing.List[typing.List[int]], tail: typing.List[int]) -> float
:canonical: src.policies.other.operators.intra_route.k_opt._connection_cost

```{autodoc2-docstring} src.policies.other.operators.intra_route.k_opt._connection_cost
```
````
