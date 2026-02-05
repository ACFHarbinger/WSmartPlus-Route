# {py:mod}`src.policies.aco_aux.pheromones`

```{py:module} src.policies.aco_aux.pheromones
```

```{autodoc2-docstring} src.policies.aco_aux.pheromones
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SparsePheromoneTau <src.policies.aco_aux.pheromones.SparsePheromoneTau>`
  - ```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau
    :summary:
    ```
````

### API

`````{py:class} SparsePheromoneTau(n_nodes: int, k: int, tau_0: float, tau_min: float, tau_max: float)
:canonical: src.policies.aco_aux.pheromones.SparsePheromoneTau

```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau.__init__
```

````{py:method} get(i: int, j: int) -> float
:canonical: src.policies.aco_aux.pheromones.SparsePheromoneTau.get

```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau.get
```

````

````{py:method} set(i: int, j: int, value: float) -> None
:canonical: src.policies.aco_aux.pheromones.SparsePheromoneTau.set

```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau.set
```

````

````{py:method} update_edge(i: int, j: int, delta: float, evaporate: bool = True) -> None
:canonical: src.policies.aco_aux.pheromones.SparsePheromoneTau.update_edge

```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau.update_edge
```

````

````{py:method} evaporate_all(rho: float) -> None
:canonical: src.policies.aco_aux.pheromones.SparsePheromoneTau.evaporate_all

```{autodoc2-docstring} src.policies.aco_aux.pheromones.SparsePheromoneTau.evaporate_all
```

````

`````
