# {py:mod}`src.policies.augmented_hybrid_volleyball_premier_league.ahvpl`

```{py:module} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl
```

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AHVPLSolver <src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver>`
  - ```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver
    :summary:
    ```
````

### API

`````{py:class} AHVPLSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.augmented_hybrid_volleyball_premier_league.params.AHVPLParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver.__init__
```

````{py:method} _initialize_population() -> typing.List[src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._initialize_population

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._initialize_population
```

````

````{py:method} _construct_individual() -> typing.Optional[src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._construct_individual

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._construct_individual
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver.solve

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver.solve
```

````

````{py:method} _select_parents(population: typing.List[src.policies.hybrid_genetic_search.individual.Individual]) -> typing.Tuple[src.policies.hybrid_genetic_search.individual.Individual, src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._select_parents

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._select_parents
```

````

````{py:method} _active_crossover(p1: src.policies.hybrid_genetic_search.individual.Individual, p2: src.policies.hybrid_genetic_search.individual.Individual) -> src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._active_crossover

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._active_crossover
```

````

````{py:method} _mutate(ind: src.policies.hybrid_genetic_search.individual.Individual) -> None
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._mutate

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._mutate
```

````

````{py:method} _alns_coaching(ind: src.policies.hybrid_genetic_search.individual.Individual, iterations: int = 100) -> src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._alns_coaching

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._alns_coaching
```

````

````{py:method} _update_pheromones(routes: typing.List[typing.List[int]], profit: float, cost: float) -> None
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._update_pheromones

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._update_pheromones
```

````

````{py:method} _routes_to_giant_tour(routes: typing.List[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._routes_to_giant_tour
:staticmethod:

```{autodoc2-docstring} src.policies.augmented_hybrid_volleyball_premier_league.ahvpl.AHVPLSolver._routes_to_giant_tour
```

````

`````
