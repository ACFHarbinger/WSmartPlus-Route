# {py:mod}`src.policies.memetic_algorithm_tolerance_selection.solver`

```{py:module} src.policies.memetic_algorithm_tolerance_selection.solver
```

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MemeticAlgorithmToleranceBasedSelectionSolver <src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver>`
  - ```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver
    :summary:
    ```
````

### API

`````{py:class} MemeticAlgorithmToleranceBasedSelectionSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.memetic_algorithm_tolerance_selection.params.MemeticAlgorithmToleranceBasedSelectionParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver

Bases: {py:obj}`logic.src.tracking.viz_mixin.PolicyVizMixin`

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver.solve

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver.solve
```

````

````{py:method} _build_random_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._build_random_solution

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._build_random_solution
```

````

````{py:method} _perturb(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._perturb

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._perturb
```

````

````{py:method} _crossover(loser_routes: typing.List[typing.List[int]], winner_routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._crossover

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._crossover
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._evaluate

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._cost

```{autodoc2-docstring} src.policies.memetic_algorithm_tolerance_selection.solver.MemeticAlgorithmToleranceBasedSelectionSolver._cost
```

````

`````
