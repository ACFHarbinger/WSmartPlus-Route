# {py:mod}`src.policies.helpers.local_search.local_search_base`

```{py:module} src.policies.helpers.local_search.local_search_base
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalSearch <src.policies.helpers.local_search.local_search_base.LocalSearch>`
  - ```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch
    :summary:
    ```
````

### API

`````{py:class} LocalSearch(dist_matrix: numpy.ndarray, waste: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, neighbors: typing.Optional[typing.Dict[int, typing.List[int]]] = None, penalty_capacity: float = 1.0, acceptance_criterion: typing.Optional[logic.src.interfaces.IAcceptanceCriterion] = None)
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch.__init__
```

````{py:method} optimize(solution: typing.Any) -> typing.Any
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch.optimize
:abstractmethod:

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch.optimize
```

````

````{py:method} _optimize_internal(target_neighborhood: typing.Optional[str] = None, active_nodes: typing.Optional[typing.Set[int]] = None, initial_profit: typing.Optional[float] = None)
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._optimize_internal

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._optimize_internal
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._calc_load_fresh

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._calc_load_fresh
```

````

````{py:method} _compute_top_insertions(route_idx: typing.Optional[int] = None)
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._compute_top_insertions

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._compute_top_insertions
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._cost

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._cost
```

````

````{py:method} _should_try_operator(op_name: str) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._should_try_operator

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._should_try_operator
```

````

````{py:method} _polar_sector(route_idx: int) -> typing.Optional[typing.Tuple[float, float]]
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._polar_sector

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._polar_sector
```

````

````{py:method} accept_move(current_obj: float, candidate_obj: float, **kwargs: typing.Any) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch.accept_move

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch.accept_move
```

````

````{py:method} step_move(current_obj: float, candidate_obj: float, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch.step_move

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch.step_move
```

````

````{py:method} _sectors_overlap(s1: typing.Optional[typing.Tuple[float, float]], s2: typing.Optional[typing.Tuple[float, float]]) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._sectors_overlap

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._sectors_overlap
```

````

````{py:method} _process_pair(u: int, v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._process_pair

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._process_pair
```

````

````{py:method} _process_inter_route(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._process_inter_route

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._process_inter_route
```

````

````{py:method} _process_intra_route(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._process_intra_route

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._process_intra_route
```

````

````{py:method} _update_map(affected_indices: typing.Set[int])
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._update_map

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._update_map
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._get_load_cached

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._get_load_cached
```

````

````{py:method} _move_relocate(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_relocate

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_relocate
```

````

````{py:method} _move_swap(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap
```

````

````{py:method} _move_swap_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap_star

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap_star
```

````

````{py:method} _move_3opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int, rng: random.Random) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_3opt_intra

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_3opt_intra
```

````

````{py:method} _move_2opt_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_2opt_star

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_2opt_star
```

````

````{py:method} _move_2opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_2opt_intra

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_2opt_intra
```

````

````{py:method} _move_or_opt(r_idx: int, pos: int, chain_len: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_or_opt

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_or_opt
```

````

````{py:method} _move_cross(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_cross

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_cross
```

````

````{py:method} _move_shift_2_0(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_shift_2_0

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_shift_2_0
```

````

````{py:method} _move_swap_2_1(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap_2_1

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap_2_1
```

````

````{py:method} _move_swap_2_2(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap_2_2

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_swap_2_2
```

````

````{py:method} _try_cross_exchange(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._try_cross_exchange

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._try_cross_exchange
```

````

````{py:method} _try_improved_cross_exchange(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._try_improved_cross_exchange

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._try_improved_cross_exchange
```

````

````{py:method} _try_lambda_interchange(r_u: int, r_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._try_lambda_interchange

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._try_lambda_interchange
```

````

````{py:method} _try_cyclic_transfer(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._try_cyclic_transfer

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._try_cyclic_transfer
```

````

````{py:method} _try_exchange_chains(r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._try_exchange_chains

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._try_exchange_chains
```

````

````{py:method} _try_ejection_chain(r_u: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._try_ejection_chain

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._try_ejection_chain
```

````

````{py:method} _move_relocate_chain(u: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_relocate_chain

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_relocate_chain
```

````

````{py:method} _move_three_permutation(u: int, r_u: int, p_u: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_three_permutation

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_three_permutation
```

````

````{py:method} _move_unrouted_insert(node: int, route_idx: int, _hint_pos: int) -> bool
:canonical: src.policies.helpers.local_search.local_search_base.LocalSearch._move_unrouted_insert

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_base.LocalSearch._move_unrouted_insert
```

````

`````
