# {py:mod}`src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl`

```{py:module} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl
```

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLAHVPLSolver <src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver
    :summary:
    ```
````

### API

`````{py:class} RLAHVPLSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.params.RLAHVPLParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver.__init__
```

````{py:method} _initialize_population() -> typing.List[src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._initialize_population

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._initialize_population
```

````

````{py:method} _construct_individual() -> typing.Optional[src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._construct_individual

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._construct_individual
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver.solve

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver.solve
```

````

````{py:method} _gls_coaching(ind: src.policies.hybrid_genetic_search.individual.Individual, iterations: int = 100) -> src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._gls_coaching

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._gls_coaching
```

````

````{py:method} _alns_coaching(ind: src.policies.hybrid_genetic_search.individual.Individual, iterations: int = 100) -> src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._alns_coaching

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._alns_coaching
```

````

````{py:method} _select_parents(population: typing.List[src.policies.hybrid_genetic_search.individual.Individual]) -> typing.Tuple[src.policies.hybrid_genetic_search.individual.Individual, src.policies.hybrid_genetic_search.individual.Individual]
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._select_parents

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._select_parents
```

````

````{py:method} _mutate(ind: src.policies.hybrid_genetic_search.individual.Individual) -> None
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._mutate

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._mutate
```

````

````{py:method} _hash_solution(ind: src.policies.hybrid_genetic_search.individual.Individual) -> int
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._hash_solution

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._hash_solution
```

````

````{py:method} _penalize_local_optimum_edges(routes: typing.List[typing.List[int]]) -> None
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._penalize_local_optimum_edges

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._penalize_local_optimum_edges
```

````

````{py:method} _augmented_evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._augmented_evaluate

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._augmented_evaluate
```

````

````{py:method} _update_pheromones_profit(routes: typing.List[typing.List[int]], profit: float, cost: float) -> None
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._update_pheromones_profit

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._update_pheromones_profit
```

````

````{py:method} _routes_to_giant_tour(routes: typing.List[typing.List[int]]) -> typing.List[int]
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._routes_to_giant_tour
:staticmethod:

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._routes_to_giant_tour
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._calculate_cost

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._calculate_cost
```

````

````{py:method} _evaluate_routes(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._evaluate_routes

```{autodoc2-docstring} src.policies.reinforcement_learning_augmented_hybrid_volleyball_premier_league.rl_ahvpl.RLAHVPLSolver._evaluate_routes
```

````

`````
