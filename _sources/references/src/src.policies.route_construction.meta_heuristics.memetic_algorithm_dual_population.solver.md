# {py:mod}`src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver`

```{py:module} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmDualPopulationSolver <src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmDualPopulationSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.params.MemeticAlgorithmDualPopulationParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver.solve
```

````

````{py:method} _initialize_population(pop_size: int) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._initialize_population

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._initialize_population
```

````

````{py:method} _substitution_phase(active_teams: typing.List[typing.List[typing.List[int]]], passive_teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._substitution_phase

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._substitution_phase
```

````

````{py:method} _coaching_phase(active_teams: typing.List[typing.List[typing.List[int]]]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._coaching_phase

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._coaching_phase
```

````

````{py:method} _learn_from_elite(current_team: typing.List[typing.List[int]], top1: typing.List[typing.List[int]], top2: typing.List[typing.List[int]], top3: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._learn_from_elite

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._learn_from_elite
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.memetic_algorithm_dual_population.solver.MemeticAlgorithmDualPopulationSolver._cost
```

````

`````
