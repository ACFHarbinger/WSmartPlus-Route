# {py:mod}`src.policies.other.local_search.local_search_manager`

```{py:module} src.policies.other.local_search.local_search_manager
```

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalSearchManager <src.policies.other.local_search.local_search_manager.LocalSearchManager>`
  - ```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager
    :summary:
    ```
````

### API

`````{py:class} LocalSearchManager(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, improvement_threshold: float, seed: typing.Optional[int] = None)
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.__init__
```

````{py:method} set_routes(routes: typing.List[typing.List[int]]) -> None
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.set_routes

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.set_routes
```

````

````{py:method} get_routes() -> typing.List[typing.List[int]]
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.get_routes

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.get_routes
```

````

````{py:method} _calc_load_fresh(route: typing.List[int]) -> float
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager._calc_load_fresh

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager._calc_load_fresh
```

````

````{py:method} _get_load_cached(r_idx: int) -> float
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager._get_load_cached

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager._get_load_cached
```

````

````{py:method} _update_map(route_indices: set) -> None
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager._update_map

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager._update_map
```

````

````{py:method} _invalidate_cache() -> None
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager._invalidate_cache

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager._invalidate_cache
```

````

````{py:method} or_opt(chain_len: int = 2) -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.or_opt

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.or_opt
```

````

````{py:method} cross_exchange_op(max_seg_len: int = 2) -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.cross_exchange_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.cross_exchange_op
```

````

````{py:method} improved_cross_exchange_op(max_seg_len: int = 2) -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.improved_cross_exchange_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.improved_cross_exchange_op
```

````

````{py:method} lambda_interchange_op(lambda_max: int = 2) -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.lambda_interchange_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.lambda_interchange_op
```

````

````{py:method} ejection_chain_op(max_depth: int = 3) -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.ejection_chain_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.ejection_chain_op
```

````

````{py:method} cyclic_transfer_op() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.cyclic_transfer_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.cyclic_transfer_op
```

````

````{py:method} exchange_chain_op() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.exchange_chain_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.exchange_chain_op
```

````

````{py:method} two_opt_star() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.two_opt_star

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.two_opt_star
```

````

````{py:method} swap_star() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.swap_star

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.swap_star
```

````

````{py:method} two_opt_intra() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.two_opt_intra

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.two_opt_intra
```

````

````{py:method} three_opt_intra() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.three_opt_intra

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.three_opt_intra
```

````

````{py:method} four_opt_intra() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.four_opt_intra

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.four_opt_intra
```

````

````{py:method} three_permutation_op() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.three_permutation_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.three_permutation_op
```

````

````{py:method} three_opt_star() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.three_opt_star

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.three_opt_star
```

````

````{py:method} four_opt_star() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.four_opt_star

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.four_opt_star
```

````

````{py:method} relocate() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.relocate

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.relocate
```

````

````{py:method} relocate_chain_op(max_chain_len: int = 3) -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.relocate_chain_op

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.relocate_chain_op
```

````

````{py:method} swap() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.swap

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.swap
```

````

````{py:method} lns() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.lns

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.lns
```

````

````{py:method} ges() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.ges

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.ges
```

````

````{py:method} lkh_refinement() -> bool
:canonical: src.policies.other.local_search.local_search_manager.LocalSearchManager.lkh_refinement

```{autodoc2-docstring} src.policies.other.local_search.local_search_manager.LocalSearchManager.lkh_refinement
```

````

`````
