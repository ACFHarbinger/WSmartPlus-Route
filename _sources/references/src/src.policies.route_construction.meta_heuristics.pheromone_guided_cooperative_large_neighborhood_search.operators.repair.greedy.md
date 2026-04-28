# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`greedy_insertion <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy.greedy_insertion>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy.greedy_insertion
    :summary:
    ```
````

### API

````{py:function} greedy_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: typing.Optional[float] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, cost_unit: float = 1.0, expand_pool: bool = True) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy.greedy_insertion

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.greedy.greedy_insertion
```
````
