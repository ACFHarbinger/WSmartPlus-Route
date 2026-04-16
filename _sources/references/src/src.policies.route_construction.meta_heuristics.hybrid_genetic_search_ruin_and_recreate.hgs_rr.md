# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HGSRRSolver <src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver
    :summary:
    ```
````

### API

`````{py:class} HGSRRSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver.solve
```

````

````{py:method} _select_parents(population: typing.List[logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.Individual]) -> typing.Tuple[logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.Individual, logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.Individual]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver._select_parents

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver._select_parents
```

````

````{py:method} _compute_operator_score(child: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.Individual, best_profit: float, best_cost: float, p1: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.Individual, p2: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.Individual) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver._compute_operator_score

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_genetic_search_ruin_and_recreate.hgs_rr.HGSRRSolver._compute_operator_score
```

````

`````
