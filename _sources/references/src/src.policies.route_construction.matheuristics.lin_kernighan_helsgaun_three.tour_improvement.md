# {py:mod}`src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement`

```{py:module} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_try_2opt_move <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move
    :summary:
    ```
* - {py:obj}`_try_3opt_move <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move
    :summary:
    ```
* - {py:obj}`_try_4opt_move <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move
    :summary:
    ```
* - {py:obj}`_try_5opt_move <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move
    :summary:
    ```
* - {py:obj}`__or_opt_relocation <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement.__or_opt_relocation>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement.__or_opt_relocation
    :summary:
    ```
* - {py:obj}`_try_oropt_move <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_oropt_move>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_oropt_move
    :summary:
    ```
* - {py:obj}`_dynamic_kopt_search <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._dynamic_kopt_search>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._dynamic_kopt_search
    :summary:
    ```
* - {py:obj}`_verify_and_construct <src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._verify_and_construct>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._verify_and_construct
    :summary:
    ```
````

### API

````{py:function} _try_2opt_move(curr_tour: typing.List[int], i: int, t1: int, t2: int, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool, int]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_2opt_move
```
````

````{py:function} _try_3opt_move(curr_tour: typing.List[int], i: int, j: int, t1: int, t2: int, t3: int, t4: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_3opt_move
```
````

````{py:function} _try_4opt_move(curr_tour: typing.List[int], i: int, j: int, k: int, t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_4opt_move
```
````

````{py:function} _try_5opt_move(curr_tour: typing.List[int], i: int, j: int, k: int, l: int, t1: int, t2: int, t3: int, t4: int, t5: int, t6: int, t7: int, t8: int, distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_5opt_move
```
````

````{py:function} __or_opt_relocation(seg_len, t1, t_first, t_after, t_last, t_dest, t_dest_after, d, load_state, waste, capacity, curr_tour, n_original, i, nodes_count, curr_p, curr_c)
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement.__or_opt_relocation

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement.__or_opt_relocation
```
````

````{py:function} _try_oropt_move(curr_tour: typing.List[int], t1: int, i: int, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_oropt_move

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._try_oropt_move
```
````

````{py:function} _dynamic_kopt_search(curr_tour: typing.List[int], i: int, t1: int, t2: int, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, n_original: typing.Optional[int] = None, load_state: typing.Optional[logic.src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.load_tracker.LoadState] = None, max_k: int = 5, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._dynamic_kopt_search

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._dynamic_kopt_search
```
````

````{py:function} _verify_and_construct(curr_tour: typing.List[int], t_list: typing.List[int], d: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], n_original: typing.Optional[int], curr_p: float, curr_c: float, pos: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.Optional[typing.List[int]], float, float, bool]
:canonical: src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._verify_and_construct

```{autodoc2-docstring} src.policies.route_construction.matheuristics.lin_kernighan_helsgaun_three.tour_improvement._verify_and_construct
```
````
