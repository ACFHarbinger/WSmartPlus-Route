# {py:mod}`src.policies.local_search.local_search_base`

```{py:module} src.policies.local_search.local_search_base
```

```{autodoc2-docstring} src.policies.local_search.local_search_base
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalSearch <src.policies.local_search.local_search_base.LocalSearch>`
  - ```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch
    :summary:
    ```
````

### API

`````{py:class} LocalSearch(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any)
:canonical: src.policies.local_search.local_search_base.LocalSearch

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch.__init__
```

````{py:method} optimize(solution: typing.Any) -> typing.Any
:canonical: src.policies.local_search.local_search_base.LocalSearch.optimize
:abstractmethod:

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch.optimize
```

````

````{py:method} _optimize_internal()
:canonical: src.policies.local_search.local_search_base.LocalSearch._optimize_internal

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._optimize_internal
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.local_search.local_search_base.LocalSearch._calc_load_fresh

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._calc_load_fresh
```

````

````{py:method} _process_node(u: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._process_node

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._process_node
```

````

````{py:method} _update_map(affected_indices: typing.Set[int])
:canonical: src.policies.local_search.local_search_base.LocalSearch._update_map

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._update_map
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.local_search.local_search_base.LocalSearch._get_load_cached

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._get_load_cached
```

````

````{py:method} _move_relocate(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._move_relocate

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._move_relocate
```

````

````{py:method} _move_swap(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._move_swap

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._move_swap
```

````

````{py:method} _move_swap_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._move_swap_star

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._move_swap_star
```

````

````{py:method} _move_3opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._move_3opt_intra

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._move_3opt_intra
```

````

````{py:method} _move_2opt_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._move_2opt_star

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._move_2opt_star
```

````

````{py:method} _move_2opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.local_search_base.LocalSearch._move_2opt_intra

```{autodoc2-docstring} src.policies.local_search.local_search_base.LocalSearch._move_2opt_intra
```

````

`````
