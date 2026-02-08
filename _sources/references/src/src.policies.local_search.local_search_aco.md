# {py:mod}`src.policies.local_search.local_search_aco`

```{py:module} src.policies.local_search.local_search_aco
```

```{autodoc2-docstring} src.policies.local_search.local_search_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOLocalSearch <src.policies.local_search.local_search_aco.ACOLocalSearch>`
  - ```{autodoc2-docstring} src.policies.local_search.local_search_aco.ACOLocalSearch
    :summary:
    ```
````

### API

`````{py:class} ACOLocalSearch(dist_matrix: numpy.ndarray, waste: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any)
:canonical: src.policies.local_search.local_search_aco.ACOLocalSearch

Bases: {py:obj}`src.policies.local_search.local_search_base.LocalSearch`

```{autodoc2-docstring} src.policies.local_search.local_search_aco.ACOLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_search.local_search_aco.ACOLocalSearch.__init__
```

````{py:method} optimize(solution: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.local_search.local_search_aco.ACOLocalSearch.optimize

```{autodoc2-docstring} src.policies.local_search.local_search_aco.ACOLocalSearch.optimize
```

````

`````
