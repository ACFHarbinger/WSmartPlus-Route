# {py:mod}`src.policies.local_search.local_search_hgs`

```{py:module} src.policies.local_search.local_search_hgs
```

```{autodoc2-docstring} src.policies.local_search.local_search_hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSLocalSearch <src.policies.local_search.local_search_hgs.HGSLocalSearch>`
  - ```{autodoc2-docstring} src.policies.local_search.local_search_hgs.HGSLocalSearch
    :summary:
    ```
````

### API

`````{py:class} HGSLocalSearch(dist_matrix: numpy.ndarray, waste: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hybrid_genetic_search.HGSParams)
:canonical: src.policies.local_search.local_search_hgs.HGSLocalSearch

Bases: {py:obj}`src.policies.local_search.local_search_base.LocalSearch`

```{autodoc2-docstring} src.policies.local_search.local_search_hgs.HGSLocalSearch
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.local_search.local_search_hgs.HGSLocalSearch.__init__
```

````{py:method} optimize(solution: src.policies.hybrid_genetic_search.Individual) -> src.policies.hybrid_genetic_search.Individual
:canonical: src.policies.local_search.local_search_hgs.HGSLocalSearch.optimize

```{autodoc2-docstring} src.policies.local_search.local_search_hgs.HGSLocalSearch.optimize
```

````

`````
