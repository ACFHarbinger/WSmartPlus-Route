# {py:mod}`src.policies.other.operators.heuristics._tour_improvement`

```{py:module} src.policies.other.operators.heuristics._tour_improvement
```

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_apply_kopt_via_operator <src.policies.other.operators.heuristics._tour_improvement._apply_kopt_via_operator>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._apply_kopt_via_operator
    :summary:
    ```
* - {py:obj}`_try_2opt_move <src.policies.other.operators.heuristics._tour_improvement._try_2opt_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_2opt_move
    :summary:
    ```
* - {py:obj}`_try_3opt_move <src.policies.other.operators.heuristics._tour_improvement._try_3opt_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_3opt_move
    :summary:
    ```
* - {py:obj}`_try_4opt_move <src.policies.other.operators.heuristics._tour_improvement._try_4opt_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_4opt_move
    :summary:
    ```
* - {py:obj}`_try_5opt_move <src.policies.other.operators.heuristics._tour_improvement._try_5opt_move>`
  - ```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_5opt_move
    :summary:
    ```
````

### API

````{py:function} _apply_kopt_via_operator(tour: typing.List[int], p_u: int, p_v: int, k: int, distance_matrix: numpy.ndarray, rng: random.Random) -> typing.Optional[typing.List[int]]
:canonical: src.policies.other.operators.heuristics._tour_improvement._apply_kopt_via_operator

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._apply_kopt_via_operator
```
````

````{py:function} _try_2opt_move(curr_tour: typing.List[int], i: int, t1: int, t2: int, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, rng: random.Random, pos_map: typing.Dict[int, int]) -> typing.Tuple[typing.Optional[typing.List[int]], float, bool, int]
:canonical: src.policies.other.operators.heuristics._tour_improvement._try_2opt_move

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_2opt_move
```
````

````{py:function} _try_3opt_move(curr_tour: typing.List[int], i: int, j: int, t1: int, t2: int, t3: int, t4: int, distance_matrix: numpy.ndarray, rng: random.Random) -> typing.Tuple[typing.Optional[typing.List[int]], float, bool]
:canonical: src.policies.other.operators.heuristics._tour_improvement._try_3opt_move

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_3opt_move
```
````

````{py:function} _try_4opt_move(curr_tour: typing.List[int], i: int, j: int, k: int, t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, distance_matrix: numpy.ndarray, rng: random.Random) -> typing.Tuple[typing.Optional[typing.List[int]], float, bool]
:canonical: src.policies.other.operators.heuristics._tour_improvement._try_4opt_move

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_4opt_move
```
````

````{py:function} _try_5opt_move(curr_tour: typing.List[int], i: int, j: int, k: int, l: int, t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, t7: int, t8: int, distance_matrix: numpy.ndarray, rng: random.Random) -> typing.Tuple[typing.Optional[typing.List[int]], float, bool]
:canonical: src.policies.other.operators.heuristics._tour_improvement._try_5opt_move

```{autodoc2-docstring} src.policies.other.operators.heuristics._tour_improvement._try_5opt_move
```
````
