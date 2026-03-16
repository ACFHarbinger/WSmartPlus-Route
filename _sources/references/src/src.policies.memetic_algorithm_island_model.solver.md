# {py:mod}`src.policies.memetic_algorithm_island_model.solver`

```{py:module} src.policies.memetic_algorithm_island_model.solver
```

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmIslandModelSolver <src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver>`
  - ```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmIslandModelSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.memetic_algorithm_island_model.params.MemeticAlgorithmIslandModelParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver.solve

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver.solve
```

````

````{py:method} _new_island() -> typing.List[typing.Tuple[typing.List[typing.List[int]], float]]
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._new_island

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._new_island
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._build_random_solution

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._build_random_solution
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._perturb

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._perturb
```

````

````{py:method} _recombine(loser_routes: typing.List[typing.List[int]], winner_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._recombine

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._recombine
```

````

````{py:method} _global_best(islands: typing.List[typing.List[typing.Tuple[typing.List[typing.List[int]], float]]]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._global_best

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._global_best
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._evaluate

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._cost

```{autodoc2-docstring} src.policies.memetic_algorithm_island_model.solver.MemeticAlgorithmIslandModelSolver._cost
```

````

`````
