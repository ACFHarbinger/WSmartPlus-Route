# {py:mod}`src.policies.ant_colony_optimization.k_sparse_aco.solver`

```{py:module} src.policies.ant_colony_optimization.k_sparse_aco.solver
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KSparseACOSolver <src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver
    :summary:
    ```
````

### API

`````{py:class} KSparseACOSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams)
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver.__init__
```

````{py:method} _nearest_neighbor_cost() -> float
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._nearest_neighbor_cost

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._nearest_neighbor_cost
```

````

````{py:method} _build_candidate_lists() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._build_candidate_lists

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._build_candidate_lists
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver.solve

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver.solve
```

````

````{py:method} _global_pheromone_update(best_routes: typing.List[typing.List[int]], best_cost: float) -> None
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._global_pheromone_update

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._global_pheromone_update
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._calculate_cost

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.solver.KSparseACOSolver._calculate_cost
```

````

`````
