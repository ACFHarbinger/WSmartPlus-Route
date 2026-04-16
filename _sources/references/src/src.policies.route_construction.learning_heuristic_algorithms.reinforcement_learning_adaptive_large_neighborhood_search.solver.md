# {py:mod}`src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver`

```{py:module} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RLALNSSolver <src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver
    :summary:
    ```
````

### API

`````{py:class} RLALNSSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.params.RLALNSParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver.__init__
```

````{py:method} _init_operators() -> None
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._init_operators

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._init_operators
```

````

````{py:method} _create_rl_agent()
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._create_rl_agent

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._create_rl_agent
```

````

````{py:method} solve(initial_solution: typing.Optional[typing.List[typing.List[int]]] = None) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver.solve

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver.solve
```

````

````{py:method} _apply_operators(routes: typing.List[typing.List[int]], d_idx: int, r_idx: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._apply_operators

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._apply_operators
```

````

````{py:method} _action_to_operators(action: int) -> typing.Tuple[int, int]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._action_to_operators

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._action_to_operators
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._calculate_cost

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._calculate_cost
```

````

````{py:method} _build_initial_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._build_initial_solution

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver._build_initial_solution
```

````

````{py:method} get_statistics() -> typing.Dict
:canonical: src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver.get_statistics

```{autodoc2-docstring} src.policies.route_construction.learning_heuristic_algorithms.reinforcement_learning_adaptive_large_neighborhood_search.solver.RLALNSSolver.get_statistics
```

````

`````
