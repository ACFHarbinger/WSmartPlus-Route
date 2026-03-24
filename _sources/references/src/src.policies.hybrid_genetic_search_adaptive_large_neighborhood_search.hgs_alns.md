# {py:mod}`src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns`

```{py:module} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSALNSSolver <src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver
    :summary:
    ```
````

### API

`````{py:class} HGSALNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.params.HGSALNSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver

Bases: {py:obj}`src.policies.hybrid_genetic_search.hgs.HGSSolver`

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver.solve

```{autodoc2-docstring} src.policies.hybrid_genetic_search_adaptive_large_neighborhood_search.hgs_alns.HGSALNSSolver.solve
```

````

`````
