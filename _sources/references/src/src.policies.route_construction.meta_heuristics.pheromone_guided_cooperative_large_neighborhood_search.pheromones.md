# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PheromoneTau <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau
    :summary:
    ```
````

### API

`````{py:class} PheromoneTau(n_nodes: int, k: int, tau_0: float, tau_min: float, tau_max: float)
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.__init__
```

````{py:method} get(i: int, j: int) -> float
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.get

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.get
```

````

````{py:method} set(i: int, j: int, value: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.set

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.set
```

````

````{py:method} update_edge(i: int, j: int, delta: float, evaporate: bool = True) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.update_edge

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.update_edge
```

````

````{py:method} evaporate_all(rho: float) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.evaporate_all

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau.evaporate_all
```

````

`````
