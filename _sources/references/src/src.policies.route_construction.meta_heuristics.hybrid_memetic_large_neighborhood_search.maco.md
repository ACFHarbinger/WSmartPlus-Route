# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticACOSolver <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver
    :summary:
    ```
````

### API

`````{py:class} MemeticACOSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.MACOParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver.solve
```

````

````{py:method} _global_pheromone_update(best_routes: typing.List[typing.List[int]], best_cost: float, iteration_best_routes: typing.List[typing.List[int]], iteration_best_cost: float, iteration: int, ib_phase: int) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._global_pheromone_update

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._global_pheromone_update
```

````

````{py:method} _deposit(routes: typing.List[typing.List[int]], cost: float, weight: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._deposit

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._deposit
```

````

````{py:method} _restart(best_routes: typing.List[typing.List[int]], best_cost: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._restart

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._restart
```

````

````{py:method} _update_elite_pool(routes: typing.List[typing.List[int]], cost: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._update_elite_pool

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._update_elite_pool
```

````

````{py:method} _nearest_neighbor_cost() -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._nearest_neighbor_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._nearest_neighbor_cost
```

````

````{py:method} _build_candidate_lists() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._build_candidate_lists

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._build_candidate_lists
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.maco.MemeticACOSolver._calculate_cost
```

````

`````
