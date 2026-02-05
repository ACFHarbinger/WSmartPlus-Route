# {py:mod}`src.policies.k_sparse_aco`

```{py:module} src.policies.k_sparse_aco
```

```{autodoc2-docstring} src.policies.k_sparse_aco
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`KSparseACOSolver <src.policies.k_sparse_aco.KSparseACOSolver>`
  - ```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_aco <src.policies.k_sparse_aco.run_aco>`
  - ```{autodoc2-docstring} src.policies.k_sparse_aco.run_aco
    :summary:
    ```
````

### API

`````{py:class} KSparseACOSolver(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, params: src.policies.aco_aux.params.ACOParams)
:canonical: src.policies.k_sparse_aco.KSparseACOSolver

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver.__init__
```

````{py:method} _nearest_neighbor_cost() -> float
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._nearest_neighbor_cost

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._nearest_neighbor_cost
```

````

````{py:method} _build_candidate_lists() -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._build_candidate_lists

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._build_candidate_lists
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.k_sparse_aco.KSparseACOSolver.solve

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver.solve
```

````

````{py:method} _construct_solution() -> typing.List[typing.List[int]]
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._construct_solution

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._construct_solution
```

````

````{py:method} _select_next_node(current: int, feasible: typing.List[int]) -> int
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._select_next_node

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._select_next_node
```

````

````{py:method} _local_pheromone_update(i: int, j: int) -> None
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._local_pheromone_update

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._local_pheromone_update
```

````

````{py:method} _global_pheromone_update(best_routes: typing.List[typing.List[int]], best_cost: float) -> None
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._global_pheromone_update

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._global_pheromone_update
```

````

````{py:method} _calculate_cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.k_sparse_aco.KSparseACOSolver._calculate_cost

```{autodoc2-docstring} src.policies.k_sparse_aco.KSparseACOSolver._calculate_cost
```

````

`````

````{py:function} run_aco(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], *args: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.k_sparse_aco.run_aco

```{autodoc2-docstring} src.policies.k_sparse_aco.run_aco
```
````
