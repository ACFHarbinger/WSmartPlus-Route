# {py:mod}`src.policies.hybrid_genetic_search`

```{py:module} src.policies.hybrid_genetic_search
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSSolver <src.policies.hybrid_genetic_search.HGSSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.HGSSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_hgs <src.policies.hybrid_genetic_search.run_hgs>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.run_hgs
    :summary:
    ```
````

### API

`````{py:class} HGSSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hgs_aux.types.HGSParams)
:canonical: src.policies.hybrid_genetic_search.HGSSolver

```{autodoc2-docstring} src.policies.hybrid_genetic_search.HGSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.HGSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_genetic_search.HGSSolver.solve

```{autodoc2-docstring} src.policies.hybrid_genetic_search.HGSSolver.solve
```

````

````{py:method} _select_parents(population: typing.List[src.policies.hgs_aux.types.Individual]) -> typing.Tuple[src.policies.hgs_aux.types.Individual, src.policies.hgs_aux.types.Individual]
:canonical: src.policies.hybrid_genetic_search.HGSSolver._select_parents

```{autodoc2-docstring} src.policies.hybrid_genetic_search.HGSSolver._select_parents
```

````

`````

````{py:function} run_hgs(dist_matrix, demands, capacity, R, C, values, *args)
:canonical: src.policies.hybrid_genetic_search.run_hgs

```{autodoc2-docstring} src.policies.hybrid_genetic_search.run_hgs
```
````
