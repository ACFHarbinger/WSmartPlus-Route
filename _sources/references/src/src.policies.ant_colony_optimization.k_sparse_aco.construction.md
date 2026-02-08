# {py:mod}`src.policies.ant_colony_optimization.k_sparse_aco.construction`

```{py:module} src.policies.ant_colony_optimization.k_sparse_aco.construction
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolutionConstructor <src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor
    :summary:
    ```
````

### API

`````{py:class} SolutionConstructor(dist_matrix: numpy.ndarray, demands: typing.Dict[int, float], capacity: float, pheromone: src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau, eta: numpy.ndarray, candidate_lists: typing.Dict[int, typing.List[int]], nodes: typing.List[int], params: src.policies.ant_colony_optimization.k_sparse_aco.params.ACOParams, tau_0: float)
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor.__init__
```

````{py:method} construct() -> typing.List[typing.List[int]]
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor.construct

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor.construct
```

````

````{py:method} _select_next_node(current: int, feasible: typing.List[int]) -> int
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor._select_next_node

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor._select_next_node
```

````

````{py:method} _local_pheromone_update(i: int, j: int) -> None
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor._local_pheromone_update

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.construction.SolutionConstructor._local_pheromone_update
```

````

`````
