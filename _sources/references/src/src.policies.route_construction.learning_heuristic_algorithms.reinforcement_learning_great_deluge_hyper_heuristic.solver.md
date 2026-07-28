# {py:mod}`src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver`

```{py:module} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLGDHHSolver <src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver
    :summary:
    ```
````

### API

`````{py:class} RLGDHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.params.RLGDHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver.solve

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver.solve
```

````

````{py:method} _select_llh() -> int
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._select_llh

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._select_llh
```

````

````{py:method} _apply_reward(u: float) -> float
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._apply_reward

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._apply_reward
```

````

````{py:method} _apply_punishment(u: float) -> float
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._apply_punishment

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._apply_punishment
```

````

````{py:method} _llh_relocate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_relocate

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_relocate
```

````

````{py:method} _llh_shaw(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_shaw

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_shaw
```

````

````{py:method} _llh_string(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_string

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_string
```

````

````{py:method} _llh_regret2(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_regret2

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_regret2
```

````

````{py:method} _initialize_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._initialize_solution

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._initialize_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._cost

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._cost
```

````

`````
