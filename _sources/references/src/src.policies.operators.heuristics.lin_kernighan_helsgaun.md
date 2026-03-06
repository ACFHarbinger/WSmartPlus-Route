# {py:mod}`src.policies.operators.heuristics.lin_kernighan_helsgaun`

```{py:module} src.policies.operators.heuristics.lin_kernighan_helsgaun
```

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_alpha_measures <src.policies.operators.heuristics.lin_kernighan_helsgaun.compute_alpha_measures>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.compute_alpha_measures
    :summary:
    ```
* - {py:obj}`get_candidate_set <src.policies.operators.heuristics.lin_kernighan_helsgaun.get_candidate_set>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.get_candidate_set
    :summary:
    ```
* - {py:obj}`calculate_penalty <src.policies.operators.heuristics.lin_kernighan_helsgaun.calculate_penalty>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.calculate_penalty
    :summary:
    ```
* - {py:obj}`get_score <src.policies.operators.heuristics.lin_kernighan_helsgaun.get_score>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.get_score
    :summary:
    ```
* - {py:obj}`is_better <src.policies.operators.heuristics.lin_kernighan_helsgaun.is_better>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.is_better
    :summary:
    ```
* - {py:obj}`apply_2opt_move <src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_2opt_move>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_2opt_move
    :summary:
    ```
* - {py:obj}`apply_3opt_move <src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_3opt_move>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_3opt_move
    :summary:
    ```
* - {py:obj}`double_bridge_kick <src.policies.operators.heuristics.lin_kernighan_helsgaun.double_bridge_kick>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.double_bridge_kick
    :summary:
    ```
* - {py:obj}`_initialize_tour <src.policies.operators.heuristics.lin_kernighan_helsgaun._initialize_tour>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._initialize_tour
    :summary:
    ```
* - {py:obj}`_try_2opt_move <src.policies.operators.heuristics.lin_kernighan_helsgaun._try_2opt_move>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._try_2opt_move
    :summary:
    ```
* - {py:obj}`_try_3opt_move <src.policies.operators.heuristics.lin_kernighan_helsgaun._try_3opt_move>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._try_3opt_move
    :summary:
    ```
* - {py:obj}`_improve_tour <src.policies.operators.heuristics.lin_kernighan_helsgaun._improve_tour>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._improve_tour
    :summary:
    ```
* - {py:obj}`solve_lkh <src.policies.operators.heuristics.lin_kernighan_helsgaun.solve_lkh>`
  - ```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.solve_lkh
    :summary:
    ```
````

### API

````{py:function} compute_alpha_measures(distance_matrix: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.compute_alpha_measures

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.compute_alpha_measures
```
````

````{py:function} get_candidate_set(distance_matrix: numpy.ndarray, alpha_measures: numpy.ndarray, max_candidates: int = 5) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.get_candidate_set

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.get_candidate_set
```
````

````{py:function} calculate_penalty(tour: typing.List[int], waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float]) -> float
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.calculate_penalty

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.calculate_penalty
```
````

````{py:function} get_score(tour: typing.List[int], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float]) -> typing.Tuple[float, float]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.get_score

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.get_score
```
````

````{py:function} is_better(p1: float, c1: float, p2: float, c2: float) -> bool
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.is_better

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.is_better
```
````

````{py:function} apply_2opt_move(tour: typing.List[int], i: int, j: int) -> typing.List[int]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_2opt_move

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_2opt_move
```
````

````{py:function} apply_3opt_move(tour: typing.List[int], i: int, j: int, k: int, case: int) -> typing.List[int]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_3opt_move

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.apply_3opt_move
```
````

````{py:function} double_bridge_kick(tour: typing.List[int], np_rng: numpy.random.Generator) -> typing.List[int]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.double_bridge_kick

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.double_bridge_kick
```
````

````{py:function} _initialize_tour(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun._initialize_tour

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._initialize_tour
```
````

````{py:function} _try_2opt_move(curr_tour: typing.List[int], i: int, t1: int, t2: int, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float]) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool, int]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun._try_2opt_move

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._try_2opt_move
```
````

````{py:function} _try_3opt_move(curr_tour: typing.List[int], i: int, j: int, t1: int, t2: int, t3: int, t4: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float]) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun._try_3opt_move

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._try_3opt_move
```
````

````{py:function} _improve_tour(curr_tour: typing.List[int], curr_pen: float, curr_cost: float, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float]) -> typing.Tuple[typing.List[int], float, float, bool]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun._improve_tour

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun._improve_tour
```
````

````{py:function} solve_lkh(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, max_iterations: int = 100, waste: typing.Optional[numpy.ndarray] = None, capacity: typing.Optional[float] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.operators.heuristics.lin_kernighan_helsgaun.solve_lkh

```{autodoc2-docstring} src.policies.operators.heuristics.lin_kernighan_helsgaun.solve_lkh
```
````
