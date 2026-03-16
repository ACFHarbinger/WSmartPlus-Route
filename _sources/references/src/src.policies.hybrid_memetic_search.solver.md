# {py:mod}`src.policies.hybrid_memetic_search.solver`

```{py:module} src.policies.hybrid_memetic_search.solver
```

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`HybridMemeticSearchSolver <src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver>`
  - ```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver
    :summary:
    ```
````

### API

`````{py:class} HybridMemeticSearchSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.hybrid_memetic_search.params.HybridMemeticSearchParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver.solve

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver.solve
```

````

````{py:method} _aco_initialization() -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._aco_initialization

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._aco_initialization
```

````

````{py:method} _random_construction() -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._random_construction

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._random_construction
```

````

````{py:method} _select_diverse_elite(population: typing.List[typing.List[typing.List[int]]], n_select: int) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._select_diverse_elite

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._select_diverse_elite
```

````

````{py:method} _hgs_evolution(active_teams: typing.List[typing.List[typing.List[int]]], active_profits: typing.List[float]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._hgs_evolution

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._hgs_evolution
```

````

````{py:method} _tournament_select(teams: typing.List[typing.List[typing.List[int]]], profits: typing.List[float], k: int = 3) -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._tournament_select

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._tournament_select
```

````

````{py:method} _crossover(parent1: typing.List[typing.List[int]], parent2: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._crossover

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._crossover
```

````

````{py:method} _mutate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._mutate

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._mutate
```

````

````{py:method} _selection(teams: typing.List[typing.List[typing.List[int]]], profits: typing.List[float]) -> typing.Tuple[typing.List[typing.List[typing.List[int]]], typing.List[float]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._selection

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._selection
```

````

````{py:method} _substitution_phase(active_teams: typing.List[typing.List[typing.List[int]]], passive_teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._substitution_phase

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._substitution_phase
```

````

````{py:method} _alns_coaching(teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._alns_coaching

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._alns_coaching
```

````

````{py:method} _update_pheromones(routes: typing.List[typing.List[int]], cost: float) -> None
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._update_pheromones

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._update_pheromones
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._evaluate

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._cost

```{autodoc2-docstring} src.policies.hybrid_memetic_search.solver.HybridMemeticSearchSolver._cost
```

````

`````
