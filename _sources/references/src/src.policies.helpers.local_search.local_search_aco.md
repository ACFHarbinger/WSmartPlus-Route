# {py:mod}`src.policies.helpers.local_search.local_search_aco`

```{py:module} src.policies.helpers.local_search.local_search_aco
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOLocalSearch <src.policies.helpers.local_search.local_search_aco.ACOLocalSearch>`
  - ```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco.ACOLocalSearch
    :summary:
    ```
````

### API

`````{py:class} ACOLocalSearch(node_coords: numpy.ndarray, waste: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, neighbors: typing.Optional[typing.Dict[int, typing.List[int]]] = None)
:canonical: src.policies.helpers.local_search.local_search_aco.ACOLocalSearch

Bases: {py:obj}`src.policies.helpers.local_search.local_search_base.LocalSearch`

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco.ACOLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco.ACOLocalSearch.__init__
```

````{py:method} _dist(i: int, j: int) -> float
:canonical: src.policies.helpers.local_search.local_search_aco.ACOLocalSearch._dist

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco.ACOLocalSearch._dist
```

````

````{py:method} calculate_route_cost(route: typing.List[int]) -> float
:canonical: src.policies.helpers.local_search.local_search_aco.ACOLocalSearch.calculate_route_cost

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco.ACOLocalSearch.calculate_route_cost
```

````

````{py:method} optimize(solution: typing.List[typing.List[int]], target_neighborhood: typing.Optional[str] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.local_search.local_search_aco.ACOLocalSearch.optimize

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_aco.ACOLocalSearch.optimize
```

````

`````
