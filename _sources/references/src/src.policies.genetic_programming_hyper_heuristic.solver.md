# {py:mod}`src.policies.genetic_programming_hyper_heuristic.solver`

```{py:module} src.policies.genetic_programming_hyper_heuristic.solver
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GPHHSolver <src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TrainingEnv <src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv>`
  - ```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv
    :summary:
    ```
````

### API

````{py:data} TrainingEnv
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv
:value: >
   None

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv
```

````

`````{py:class} GPHHSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.genetic_programming_hyper_heuristic.params.GPHHParams, mandatory_nodes: typing.Optional[typing.List[int]] = None, training_environments: typing.Optional[typing.List[src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv]] = None)
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver.__init__
```

````{py:method} _build_knn(dm: numpy.ndarray, nodes: typing.List[int], k: int) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_knn
:staticmethod:

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_knn
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver.solve

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver.solve
```

````

````{py:method} _construct_solution(tree: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, nodes: typing.List[int], wastes: typing.Dict[int, float], mandatory: typing.Set[int], dm: numpy.ndarray, knn: typing.Dict[int, typing.List[int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._construct_solution

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._construct_solution
```

````

````{py:method} _build_insertion_context(node_revenue: float, route: typing.List[int], node: int, insertion_cost: float, dm: numpy.ndarray, remaining_capacity: float) -> typing.Dict[str, float]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_insertion_context

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._build_insertion_context
```

````

````{py:method} _cheapest_insertion(route: typing.List[int], node: int, dm: numpy.ndarray) -> typing.Tuple[int, float]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._cheapest_insertion

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._cheapest_insertion
```

````

````{py:method} _min_distance_to_route(node: int, route: typing.List[int], dm: numpy.ndarray) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._min_distance_to_route
:staticmethod:

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._min_distance_to_route
```

````

````{py:method} _resolve_training() -> typing.List[src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv]
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._resolve_training

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._resolve_training
```

````

````{py:method} _evaluate_tree(tree: src.policies.genetic_programming_hyper_heuristic.tree.GPNode, training: typing.List[src.policies.genetic_programming_hyper_heuristic.solver.TrainingEnv]) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate_tree

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate_tree
```

````

````{py:method} _tournament(pop: typing.List[src.policies.genetic_programming_hyper_heuristic.tree.GPNode], fitness: typing.List[float]) -> src.policies.genetic_programming_hyper_heuristic.tree.GPNode
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._tournament

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._tournament
```

````

````{py:method} _evaluate_routes(routes: typing.List[typing.List[int]], wastes: typing.Dict[int, float], dm: numpy.ndarray) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate_routes

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._evaluate_routes
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]], dm: numpy.ndarray) -> float
:canonical: src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._cost

```{autodoc2-docstring} src.policies.genetic_programming_hyper_heuristic.solver.GPHHSolver._cost
```

````

`````
