# {py:mod}`src.policies.lin_kernighan_helsgaun_three.tour_improvement`

```{py:module} src.policies.lin_kernighan_helsgaun_three.tour_improvement
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_try_2opt_move <src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move
    :summary:
    ```
* - {py:obj}`_try_3opt_move <src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move
    :summary:
    ```
* - {py:obj}`_try_4opt_move <src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move
    :summary:
    ```
* - {py:obj}`_try_5opt_move <src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move
    :summary:
    ```
````

### API

````{py:function} _try_2opt_move(curr_tour: typing.List[int], i: int, t1: int, t2: int, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool, int]
:canonical: src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move
```
````

````{py:function} _try_3opt_move(curr_tour: typing.List[int], i: int, j: int, t1: int, t2: int, t3: int, t4: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move
```
````

````{py:function} _try_4opt_move(curr_tour: typing.List[int], i: int, j: int, k: int, t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move
```
````

````{py:function} _try_5opt_move(curr_tour: typing.List[int], i: int, j: int, k: int, l: int, t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, t7: int, t8: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move
```
````
