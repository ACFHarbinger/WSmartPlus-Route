# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ACOSolver <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver
    :summary:
    ```
````

### API

`````{py:class} ACOSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver.__init__
```

````{py:method} _nearest_neighbor_cost() -> float
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._nearest_neighbor_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._nearest_neighbor_cost
```

````

````{py:method} _build_candidate_lists() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._build_candidate_lists

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._build_candidate_lists
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver.solve
```

````

````{py:method} _global_pheromone_update(best_routes: typing.List[typing.List[int]], best_cost: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._global_pheromone_update

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._global_pheromone_update
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.aco.ACOSolver._calculate_cost
```

````

`````
