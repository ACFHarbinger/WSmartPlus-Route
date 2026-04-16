# {py:mod}`src.policies.helpers.local_search.local_search_filo`

```{py:module} src.policies.helpers.local_search.local_search_filo
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FILOLocalSearch <src.policies.helpers.local_search.local_search_filo.FILOLocalSearch>`
  - ```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`K_NEIGHBORS <src.policies.helpers.local_search.local_search_filo.K_NEIGHBORS>`
  - ```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.K_NEIGHBORS
    :summary:
    ```
* - {py:obj}`EJCH_MAX_DEPTH <src.policies.helpers.local_search.local_search_filo.EJCH_MAX_DEPTH>`
  - ```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.EJCH_MAX_DEPTH
    :summary:
    ```
````

### API

````{py:data} K_NEIGHBORS
:canonical: src.policies.helpers.local_search.local_search_filo.K_NEIGHBORS
:type: int
:value: >
   20

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.K_NEIGHBORS
```

````

````{py:data} EJCH_MAX_DEPTH
:canonical: src.policies.helpers.local_search.local_search_filo.EJCH_MAX_DEPTH
:type: int
:value: >
   3

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.EJCH_MAX_DEPTH
```

````

`````{py:class} FILOLocalSearch(*args: typing.Any, **kwargs: typing.Any)
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch

Bases: {py:obj}`logic.src.policies.helpers.local_search.local_search_base.LocalSearch`

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch.__init__
```

````{py:method} _rebuild_index() -> None
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._rebuild_index

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._rebuild_index
```

````

````{py:method} _rebuild_route_positions(ri: int) -> None
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._rebuild_route_positions

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._rebuild_route_positions
```

````

````{py:method} _find_node(u: int) -> typing.Tuple[int, int]
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._find_node

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._find_node
```

````

````{py:method} _get_gain_relocate(u: int, ri: int, pi: int, rj: int, pj: int) -> float
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_relocate

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_relocate
```

````

````{py:method} _get_gain_swap(u: int, ri: int, pi: int, v: int, rj: int, pj: int) -> float
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_swap

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_swap
```

````

````{py:method} _get_gain_2opt(ri: int, pi: int, pj: int) -> float
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_2opt

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_2opt
```

````

````{py:method} _get_gain_tails(ri: int, pi: int, rj: int, pj: int) -> float
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_tails

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_tails
```

````

````{py:method} _get_gain_split(ri: int, pi: int) -> float
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_split

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._get_gain_split
```

````

````{py:method} _gamma_neighbors(u: int) -> typing.List[int]
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._gamma_neighbors

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._gamma_neighbors
```

````

````{py:method} _try_relocate(active_nodes: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_relocate

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_relocate
```

````

````{py:method} _try_swap(active_nodes: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_swap

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_swap
```

````

````{py:method} _try_2opt(active_nodes: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_2opt

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_2opt
```

````

````{py:method} _try_tails(active_nodes: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_tails

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_tails
```

````

````{py:method} _try_split(active_nodes: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_split

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_split
```

````

````{py:method} _try_ejection_chain(active_nodes: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_ejection_chain

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._try_ejection_chain
```

````

````{py:method} _recursive_eject(u: int, ri: int, pi: int, depth: int, visited: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._recursive_eject

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._recursive_eject
```

````

````{py:method} _place_displaced(v: int, depth: int, visited: typing.Set[int]) -> bool
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._place_displaced

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch._place_displaced
```

````

````{py:method} optimize(solution: typing.List[typing.List[int]], active_nodes: typing.Optional[typing.Set[int]] = None, node_gamma: typing.Optional[typing.List[float]] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.local_search.local_search_filo.FILOLocalSearch.optimize

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_filo.FILOLocalSearch.optimize
```

````

`````
