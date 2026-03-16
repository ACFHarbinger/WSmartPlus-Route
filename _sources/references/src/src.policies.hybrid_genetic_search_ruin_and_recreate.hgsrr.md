# {py:mod}`src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr`

```{py:module} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSRRSolver <src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_hgsrr <src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.run_hgsrr>`
  - ```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.run_hgsrr
    :summary:
    ```
````

### API

`````{py:class} HGSRRSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hybrid_genetic_search_ruin_and_recreate.params.HGSRRParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver.solve

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver.solve
```

````

````{py:method} _select_parents(population: typing.List[logic.src.policies.hybrid_genetic_search.Individual]) -> typing.Tuple[logic.src.policies.hybrid_genetic_search.Individual, logic.src.policies.hybrid_genetic_search.Individual]
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver._select_parents

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver._select_parents
```

````

````{py:method} _compute_operator_score(child: logic.src.policies.hybrid_genetic_search.Individual, best_profit: float, best_cost: float, p1: logic.src.policies.hybrid_genetic_search.Individual, p2: logic.src.policies.hybrid_genetic_search.Individual) -> float
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver._compute_operator_score

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.HGSRRSolver._compute_operator_score
```

````

`````

````{py:function} run_hgsrr(dist_matrix, wastes, capacity, R, C, values, mandatory_nodes=None, *args)
:canonical: src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.run_hgsrr

```{autodoc2-docstring} src.policies.hybrid_genetic_search_ruin_and_recreate.hgsrr.run_hgsrr
```
````
