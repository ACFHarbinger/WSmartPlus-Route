# {py:mod}`src.policies.ant_colony_optimization.k_sparse_aco.pheromones`

```{py:module} src.policies.ant_colony_optimization.k_sparse_aco.pheromones
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparsePheromoneTau <src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau>`
  - ```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau
    :summary:
    ```
````

### API

`````{py:class} SparsePheromoneTau(n_nodes: int, k: int, tau_0: float, tau_min: float, tau_max: float)
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.__init__
```

````{py:method} get(i: int, j: int) -> float
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.get

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.get
```

````

````{py:method} set(i: int, j: int, value: float) -> None
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.set

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.set
```

````

````{py:method} update_edge(i: int, j: int, delta: float, evaporate: bool = True) -> None
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.update_edge

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.update_edge
```

````

````{py:method} evaporate_all(rho: float) -> None
:canonical: src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.evaporate_all

```{autodoc2-docstring} src.policies.ant_colony_optimization.k_sparse_aco.pheromones.SparsePheromoneTau.evaporate_all
```

````

`````
