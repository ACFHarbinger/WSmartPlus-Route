# {py:mod}`src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl`

```{py:module} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl
```

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLHVPLSolver <src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver
    :summary:
    ```
````

### API

`````{py:class} RLHVPLSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.params.RLHVPLParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver.solve

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver.solve
```

````

````{py:method} _apply_coaching(routes: typing.List[typing.List[int]], iterations: int) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._apply_coaching

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._apply_coaching
```

````

````{py:method} _canonicalize_routes(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._canonicalize_routes

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._canonicalize_routes
```

````

````{py:method} _hash_routes(routes: typing.List[typing.List[int]]) -> str
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._hash_routes

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._hash_routes
```

````

````{py:method} _get_best(population: typing.List[typing.Tuple[typing.List[typing.List[int]], float, float]]) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._get_best

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._get_best
```

````

````{py:method} _update_pheromones(routes: typing.List[typing.List[int]], profit: float, cost: float) -> None
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._update_pheromones

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._update_pheromones
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._calculate_cost

```{autodoc2-docstring} src.policies.reinforcement_learning_hybrid_volleyball_premier_league.rl_hvpl.RLHVPLSolver._calculate_cost
```

````

`````
