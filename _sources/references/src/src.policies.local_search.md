# {py:mod}`src.policies.local_search`

```{py:module} src.policies.local_search
```

```{autodoc2-docstring} src.policies.local_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LocalSearch <src.policies.local_search.LocalSearch>`
  - ```{autodoc2-docstring} src.policies.local_search.LocalSearch
    :summary:
    ```
* - {py:obj}`HGSLocalSearch <src.policies.local_search.HGSLocalSearch>`
  - ```{autodoc2-docstring} src.policies.local_search.HGSLocalSearch
    :summary:
    ```
* - {py:obj}`ACOLocalSearch <src.policies.local_search.ACOLocalSearch>`
  - ```{autodoc2-docstring} src.policies.local_search.ACOLocalSearch
    :summary:
    ```
````

### API

`````{py:class} LocalSearch(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any)
:canonical: src.policies.local_search.LocalSearch

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.local_search.LocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_search.LocalSearch.__init__
```

````{py:method} optimize(solution: typing.Any) -> typing.Any
:canonical: src.policies.local_search.LocalSearch.optimize
:abstractmethod:

```{autodoc2-docstring} src.policies.local_search.LocalSearch.optimize
```

````

````{py:method} _optimize_internal()
:canonical: src.policies.local_search.LocalSearch._optimize_internal

```{autodoc2-docstring} src.policies.local_search.LocalSearch._optimize_internal
```

````

````{py:method} _calc_load_fresh(r: typing.List[int]) -> float
:canonical: src.policies.local_search.LocalSearch._calc_load_fresh

```{autodoc2-docstring} src.policies.local_search.LocalSearch._calc_load_fresh
```

````

````{py:method} _process_node(u: int) -> bool
:canonical: src.policies.local_search.LocalSearch._process_node

```{autodoc2-docstring} src.policies.local_search.LocalSearch._process_node
```

````

````{py:method} _update_map(affected_indices: typing.Set[int])
:canonical: src.policies.local_search.LocalSearch._update_map

```{autodoc2-docstring} src.policies.local_search.LocalSearch._update_map
```

````

````{py:method} _get_load_cached(ri: int) -> float
:canonical: src.policies.local_search.LocalSearch._get_load_cached

```{autodoc2-docstring} src.policies.local_search.LocalSearch._get_load_cached
```

````

````{py:method} _move_relocate(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.LocalSearch._move_relocate

```{autodoc2-docstring} src.policies.local_search.LocalSearch._move_relocate
```

````

````{py:method} _move_swap(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.LocalSearch._move_swap

```{autodoc2-docstring} src.policies.local_search.LocalSearch._move_swap
```

````

````{py:method} _move_swap_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.LocalSearch._move_swap_star

```{autodoc2-docstring} src.policies.local_search.LocalSearch._move_swap_star
```

````

````{py:method} _move_3opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.LocalSearch._move_3opt_intra

```{autodoc2-docstring} src.policies.local_search.LocalSearch._move_3opt_intra
```

````

````{py:method} _move_2opt_star(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.LocalSearch._move_2opt_star

```{autodoc2-docstring} src.policies.local_search.LocalSearch._move_2opt_star
```

````

````{py:method} _move_2opt_intra(u: int, v: int, r_u: int, p_u: int, r_v: int, p_v: int) -> bool
:canonical: src.policies.local_search.LocalSearch._move_2opt_intra

```{autodoc2-docstring} src.policies.local_search.LocalSearch._move_2opt_intra
```

````

`````

`````{py:class} HGSLocalSearch(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hgs_aux.types.HGSParams)
:canonical: src.policies.local_search.HGSLocalSearch

Bases: {py:obj}`src.policies.local_search.LocalSearch`

```{autodoc2-docstring} src.policies.local_search.HGSLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_search.HGSLocalSearch.__init__
```

````{py:method} optimize(solution: src.policies.hgs_aux.types.Individual) -> src.policies.hgs_aux.types.Individual
:canonical: src.policies.local_search.HGSLocalSearch.optimize

```{autodoc2-docstring} src.policies.local_search.HGSLocalSearch.optimize
```

````

`````

`````{py:class} ACOLocalSearch(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any)
:canonical: src.policies.local_search.ACOLocalSearch

Bases: {py:obj}`src.policies.local_search.LocalSearch`

```{autodoc2-docstring} src.policies.local_search.ACOLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_search.ACOLocalSearch.__init__
```

````{py:method} optimize(solution: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.local_search.ACOLocalSearch.optimize

```{autodoc2-docstring} src.policies.local_search.ACOLocalSearch.optimize
```

````

`````
