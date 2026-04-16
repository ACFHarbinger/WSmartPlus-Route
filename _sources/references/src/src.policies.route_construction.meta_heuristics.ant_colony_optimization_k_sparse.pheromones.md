# {py:mod}`src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones`

```{py:module} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparsePheromoneTau <src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau
    :summary:
    ```
````

### API

`````{py:class} SparsePheromoneTau(n_nodes: int, tau_0: float, scale: float, tau_min: float, tau_max: float)
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.__init__
```

````{py:method} get(i: int, j: int) -> float
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.get

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.get
```

````

````{py:method} set(i: int, j: int, value: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.set

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.set
```

````

````{py:method} deposit_edge(i: int, j: int, delta: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.deposit_edge

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.deposit_edge
```

````

````{py:method} evaporate_all(rho: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.evaporate_all

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau.evaporate_all
```

````

`````
