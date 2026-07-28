# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PGCLNSSolver <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver
    :summary:
    ```
````

### API

`````{py:class} PGCLNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.PGCLNSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver.solve
```

````

````{py:method} _canonicalize_routes(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._canonicalize_routes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._canonicalize_routes
```

````

````{py:method} _hash_routes(routes: typing.List[typing.List[int]]) -> str
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._hash_routes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._hash_routes
```

````

````{py:method} _get_best(population: typing.List[typing.Tuple[typing.List[typing.List[int]], float, float]]) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._get_best

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._get_best
```

````

````{py:method} _update_pheromones(routes: typing.List[typing.List[int]], cost: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._update_pheromones

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._update_pheromones
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pg_clns.PGCLNSSolver._calculate_cost
```

````

`````
