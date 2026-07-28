# {py:mod}`src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones`

```{py:module} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparsePheromoneTau <src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau
    :summary:
    ```
````

### API

`````{py:class} SparsePheromoneTau(n_nodes: int, tau_0: float, scale: float, rho: float, *, tau_min_fixed: typing.Optional[float] = None, tau_max_fixed: typing.Optional[float] = None, p_best: float = 0.05, avg_candidates: float = 8.0)
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.__init__
```

````{py:method} _compute_tau_min() -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau._compute_tau_min

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau._compute_tau_min
```

````

````{py:method} update_bounds(best_cost: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.update_bounds

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.update_bounds
```

````

````{py:method} get(i: int, j: int) -> float
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.get

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.get
```

````

````{py:method} set(i: int, j: int, value: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.set

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.set
```

````

````{py:method} deposit_edge(i: int, j: int, delta: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.deposit_edge

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.deposit_edge
```

````

````{py:method} evaporate_all(rho: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.evaporate_all

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.evaporate_all
```

````

````{py:method} reinitialize(tau_reset: typing.Optional[float] = None) -> None
:canonical: src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.reinitialize

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.hybrid_memetic_large_neighborhood_search.pheromones.SparsePheromoneTau.reinitialize
```

````

`````
