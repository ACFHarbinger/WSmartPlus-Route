# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SolutionConstructor <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor
    :summary:
    ```
````

### API

`````{py:class} SolutionConstructor(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, pheromone: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.pheromones.PheromoneTau, eta: numpy.ndarray, candidate_lists: typing.Dict[int, typing.List[int]], nodes: typing.List[int], params: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.params.ACOParams, tau_0: float, R: float = 0.0, C: float = 1.0, mandatory_nodes: typing.Optional[typing.List[int]] = None, seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor.__init__
```

````{py:method} construct() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor.construct

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor.construct
```

````

````{py:method} _select_next_node(current: int, feasible: typing.List[int]) -> int
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor._select_next_node

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor._select_next_node
```

````

````{py:method} _local_pheromone_update(i: int, j: int) -> None
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor._local_pheromone_update

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.construction.SolutionConstructor._local_pheromone_update
```

````

`````
