# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridMemeticLargeNeighborhoodSearchSolver <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver
    :summary:
    ```
````

### API

`````{py:class} HybridMemeticLargeNeighborhoodSearchSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.params.HybridMemeticLargeNeighborhoodSearchParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver.solve
```

````

````{py:method} _maco_initialization() -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._maco_initialization

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._maco_initialization
```

````

````{py:method} _greedy_construction() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._greedy_construction

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._greedy_construction
```

````

````{py:method} _select_diverse_elite(population: typing.List[typing.List[typing.List[int]]], n_select: int) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._select_diverse_elite

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._select_diverse_elite
```

````

````{py:method} _hgs_evolution(active_teams: typing.List[typing.List[typing.List[int]]], active_profits: typing.List[float]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._hgs_evolution

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._hgs_evolution
```

````

````{py:method} _tournament_select(teams: typing.List[typing.List[typing.List[int]]], profits: typing.List[float], k: int = 3) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._tournament_select

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._tournament_select
```

````

````{py:method} _crossover(parent1: typing.List[typing.List[int]], parent2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._crossover

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._crossover
```

````

````{py:method} _mutate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._mutate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._mutate
```

````

````{py:method} _selection(teams: typing.List[typing.List[typing.List[int]]], profits: typing.List[float]) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[float]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._selection

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._selection
```

````

````{py:method} _substitution_phase(active_teams: typing.List[typing.List[typing.List[int]]], passive_teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._substitution_phase

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._substitution_phase
```

````

````{py:method} _alns_coaching(teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._alns_coaching

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._alns_coaching
```

````

````{py:method} _update_pheromones(routes: typing.List[typing.List[int]], cost: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._update_pheromones

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._update_pheromones
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.solver.HybridMemeticLargeNeighborhoodSearchSolver._cost
```

````

`````
