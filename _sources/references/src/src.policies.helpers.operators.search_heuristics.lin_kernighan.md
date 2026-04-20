# {py:mod}`src.policies.helpers.operators.search_heuristics.lin_kernighan`

```{py:module} src.policies.helpers.operators.search_heuristics.lin_kernighan
```

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_build_pos_map <src.policies.helpers.operators.search_heuristics.lin_kernighan._build_pos_map>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._build_pos_map
    :summary:
    ```
* - {py:obj}`_apply_2opt_positions <src.policies.helpers.operators.search_heuristics.lin_kernighan._apply_2opt_positions>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._apply_2opt_positions
    :summary:
    ```
* - {py:obj}`_succ <src.policies.helpers.operators.search_heuristics.lin_kernighan._succ>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._succ
    :summary:
    ```
* - {py:obj}`_pred <src.policies.helpers.operators.search_heuristics.lin_kernighan._pred>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._pred
    :summary:
    ```
* - {py:obj}`_lk_search <src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_search>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_search
    :summary:
    ```
* - {py:obj}`_lk_extend <src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_extend>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_extend
    :summary:
    ```
* - {py:obj}`_improve_tour_lk <src.policies.helpers.operators.search_heuristics.lin_kernighan._improve_tour_lk>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._improve_tour_lk
    :summary:
    ```
* - {py:obj}`solve_lk <src.policies.helpers.operators.search_heuristics.lin_kernighan.solve_lk>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan.solve_lk
    :summary:
    ```
````

### API

````{py:function} _build_pos_map(tour: typing.List[int]) -> numpy.ndarray
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._build_pos_map

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._build_pos_map
```
````

````{py:function} _apply_2opt_positions(tour: typing.List[int], pos: numpy.ndarray, p1: int, p2: int) -> typing.Tuple[typing.List[int], numpy.ndarray]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._apply_2opt_positions

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._apply_2opt_positions
```
````

````{py:function} _succ(p: int, n: int) -> int
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._succ

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._succ
```
````

````{py:function} _pred(p: int, n: int) -> int
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._pred

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._pred
```
````

````{py:function} _lk_search(t1: int, tour: typing.List[int], pos: numpy.ndarray, n: int, d: numpy.ndarray, candidates: typing.Dict[int, typing.List[int]], max_depth: int) -> typing.Optional[typing.Tuple[typing.List[int], numpy.ndarray, float]]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_search

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_search
```
````

````{py:function} _lk_extend(t1: int, t_free: int, p_free: int, G: float, depth: int, tour: typing.List[int], pos: numpy.ndarray, n: int, d: numpy.ndarray, candidates: typing.Dict[int, typing.List[int]], max_depth: int, broken: set, added: set) -> typing.Optional[typing.Tuple[typing.List[int], numpy.ndarray, float]]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_extend

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._lk_extend
```
````

````{py:function} _improve_tour_lk(tour: typing.List[int], cost: float, candidates: typing.Dict[int, typing.List[int]], d: numpy.ndarray, dont_look: numpy.ndarray, max_depth: int) -> typing.Tuple[typing.List[int], float, bool, numpy.ndarray]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan._improve_tour_lk

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan._improve_tour_lk
```
````

````{py:function} solve_lk(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 50, max_depth: int = 5, n_candidates: int = 5, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, seed: typing.Optional[int] = None) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.helpers.operators.search_heuristics.lin_kernighan.solve_lk

```{autodoc2-docstring} src.policies.helpers.operators.search_heuristics.lin_kernighan.solve_lk
```
````
