# {py:mod}`src.policies.other.reinforcement_learning.ks_aco_qlearning`

```{py:module} src.policies.other.reinforcement_learning.ks_aco_qlearning
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KSparseACOQLSolver <src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver>`
  - ```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver
    :summary:
    ```
````

### API

`````{py:class} KSparseACOQLSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.ant_colony_optimization_k_sparse.params.KSACOParams, rl_params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver.__init__
```

````{py:method} _nearest_neighbor_cost() -> float
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._nearest_neighbor_cost

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._nearest_neighbor_cost
```

````

````{py:method} _build_candidate_lists() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._build_candidate_lists

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._build_candidate_lists
```

````

````{py:method} _initialize_with_nn_heuristic() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._initialize_with_nn_heuristic

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._initialize_with_nn_heuristic
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver.solve

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver.solve
```

````

````{py:method} _q_learning_local_search(routes: typing.List[typing.List[int]], iteration: int) -> typing.List[typing.List[int]]
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._q_learning_local_search

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._q_learning_local_search
```

````

````{py:method} _apply_operator(operator_name: str) -> bool
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._apply_operator

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._apply_operator
```

````

````{py:method} _global_pheromone_update(best_routes: typing.List[typing.List[int]], best_cost: float)
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._global_pheromone_update

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._global_pheromone_update
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._calculate_cost

```{autodoc2-docstring} src.policies.other.reinforcement_learning.ks_aco_qlearning.KSparseACOQLSolver._calculate_cost
```

````

`````
