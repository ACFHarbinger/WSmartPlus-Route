# {py:mod}`src.policies.hybrid_genetic_search.hgs`

```{py:module} src.policies.hybrid_genetic_search.hgs
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSSolver <src.policies.hybrid_genetic_search.hgs.HGSSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.HGSSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_hgs <src.policies.hybrid_genetic_search.hgs.run_hgs>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.run_hgs
    :summary:
    ```
````

### API

`````{py:class} HGSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hybrid_genetic_search.params.HGSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.hybrid_genetic_search.hgs.HGSSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.HGSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.HGSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_genetic_search.hgs.HGSSolver.solve

```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.HGSSolver.solve
```

````

````{py:method} _select_parents(population: typing.List[src.policies.hybrid_genetic_search.individual.Individual]) -> typing.Tuple[src.policies.hybrid_genetic_search.individual.Individual, src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.hybrid_genetic_search.hgs.HGSSolver._select_parents

```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.HGSSolver._select_parents
```

````

`````

````{py:function} run_hgs(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args)
:canonical: src.policies.hybrid_genetic_search.hgs.run_hgs

```{autodoc2-docstring} src.policies.hybrid_genetic_search.hgs.run_hgs
```
````
