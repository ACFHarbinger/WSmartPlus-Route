# {py:mod}`src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction`

```{py:module} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolutionConstructor <src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor
    :summary:
    ```
````

### API

`````{py:class} SolutionConstructor(node_coords: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, pheromone: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.pheromones.SparsePheromoneTau, candidate_lists: typing.Dict[int, typing.List[int]], nodes: typing.List[int], params: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.params.KSACOParams, tau_0: float, R: float = 0.0, C: float = 1.0, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor.__init__
```

````{py:method} construct() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor.construct

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor.construct
```

````

````{py:method} _cleanup_unvisited(unvisited: typing.Set[int], mandatory_unvisited: typing.Set[int]) -> None
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._cleanup_unvisited

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._cleanup_unvisited
```

````

````{py:method} _dist(i: int, j: int) -> numpy.floating[typing.Any]
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._dist

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._dist
```

````

````{py:method} _any_profitable_nodes(unvisited: typing.Set[int]) -> bool
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._any_profitable_nodes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._any_profitable_nodes
```

````

````{py:method} _get_feasible_nodes(unvisited: typing.Set[int], mandatory_unvisited: typing.Set[int], load: float, current: int) -> typing.List[int]
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._get_feasible_nodes

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._get_feasible_nodes
```

````

````{py:method} _select_next_node(current: int, feasible: typing.List[int]) -> int
:canonical: src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._select_next_node

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.ant_colony_optimization_k_sparse.construction.SolutionConstructor._select_next_node
```

````

`````
