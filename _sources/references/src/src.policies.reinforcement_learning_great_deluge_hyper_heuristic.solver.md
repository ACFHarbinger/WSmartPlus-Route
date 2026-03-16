# {py:mod}`src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver`

```{py:module} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLGDHHSolver <src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver>`
  - ```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver
    :summary:
    ```
````

### API

`````{py:class} RLGDHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.params.RLGDHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver.solve

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver.solve
```

````

````{py:method} _select_heuristic() -> int
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._select_heuristic

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._select_heuristic
```

````

````{py:method} _llh_swap(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_swap

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_swap
```

````

````{py:method} _llh_relocate(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_relocate

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_relocate
```

````

````{py:method} _llh_2opt(routes: typing.List[typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_2opt

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._llh_2opt
```

````

````{py:method} _initialize_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._initialize_solution

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._initialize_solution
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._evaluate

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._cost

```{autodoc2-docstring} src.policies.reinforcement_learning_great_deluge_hyper_heuristic.solver.RLGDHHSolver._cost
```

````

`````
