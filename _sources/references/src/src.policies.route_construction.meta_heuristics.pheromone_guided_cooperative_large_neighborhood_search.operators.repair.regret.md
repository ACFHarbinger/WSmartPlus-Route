# {py:mod}`src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret`

```{py:module} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`regret_2_insertion <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_2_insertion>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_2_insertion
    :summary:
    ```
* - {py:obj}`regret_k_insertion <src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_k_insertion>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_k_insertion
    :summary:
    ```
````

### API

````{py:function} regret_2_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: typing.Optional[float] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, cost_unit: float = 1.0) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_2_insertion

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_2_insertion
```
````

````{py:function} regret_k_insertion(routes: typing.List[typing.List[int]], removed_nodes: typing.List[int], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, k: int = 2, R: typing.Optional[float] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, cost_unit: float = 1.0) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_k_insertion

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.pheromone_guided_cooperative_large_neighborhood_search.operators.repair.regret.regret_k_insertion
```
````
