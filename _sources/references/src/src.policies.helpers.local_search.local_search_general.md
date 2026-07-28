# {py:mod}`src.policies.helpers.local_search.local_search_general`

```{py:module} src.policies.helpers.local_search.local_search_general
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_general
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GeneralLocalSearch <src.policies.helpers.local_search.local_search_general.GeneralLocalSearch>`
  - ```{autodoc2-docstring} src.policies.helpers.local_search.local_search_general.GeneralLocalSearch
    :summary:
    ```
````

### API

`````{py:class} GeneralLocalSearch(dist_matrix: numpy.ndarray, waste: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, neighbors: typing.Optional[typing.Dict[int, typing.List[int]]] = None)
:canonical: src.policies.helpers.local_search.local_search_general.GeneralLocalSearch

Bases: {py:obj}`src.policies.helpers.local_search.local_search_base.LocalSearch`

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_general.GeneralLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_general.GeneralLocalSearch.__init__
```

````{py:method} optimize(solution: typing.List[typing.List[int]], target_neighborhood: typing.Optional[str] = None) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.local_search.local_search_general.GeneralLocalSearch.optimize

```{autodoc2-docstring} src.policies.helpers.local_search.local_search_general.GeneralLocalSearch.optimize
```

````

`````
