# {py:mod}`src.policies.hgs_aux.local_search`

```{py:module} src.policies.hgs_aux.local_search
```

```{autodoc2-docstring} src.policies.hgs_aux.local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalSearch <src.policies.hgs_aux.local_search.LocalSearch>`
  - ```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch
    :summary:
    ```
````

### API

`````{py:class} LocalSearch(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hgs_aux.types.HGSParams)
:canonical: src.policies.hgs_aux.local_search.LocalSearch

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch.__init__
```

````{py:method} optimize(individual: src.policies.hgs_aux.types.Individual) -> src.policies.hgs_aux.types.Individual
:canonical: src.policies.hgs_aux.local_search.LocalSearch.optimize

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch.optimize
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.hgs_aux.local_search.LocalSearch._calc_load_fresh

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._calc_load_fresh
```

````

````{py:method} _process_node(u: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._process_node

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._process_node
```

````

````{py:method} _update_map(affected_indices: typing.Set[int])
:canonical: src.policies.hgs_aux.local_search.LocalSearch._update_map

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._update_map
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.hgs_aux.local_search.LocalSearch._get_load_cached

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._get_load_cached
```

````

````{py:method} _move_relocate(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._move_relocate

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._move_relocate
```

````

````{py:method} _move_swap(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._move_swap

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._move_swap
```

````

````{py:method} _move_swap_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._move_swap_star

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._move_swap_star
```

````

````{py:method} _move_3opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._move_3opt_intra

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._move_3opt_intra
```

````

````{py:method} _move_2opt_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._move_2opt_star

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._move_2opt_star
```

````

````{py:method} _move_2opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.hgs_aux.local_search.LocalSearch._move_2opt_intra

```{autodoc2-docstring} src.policies.hgs_aux.local_search.LocalSearch._move_2opt_intra
```

````

`````
