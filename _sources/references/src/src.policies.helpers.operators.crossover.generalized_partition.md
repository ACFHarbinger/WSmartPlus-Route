# {py:mod}`src.policies.helpers.operators.crossover.generalized_partition`

```{py:module} src.policies.helpers.operators.crossover.generalized_partition
```

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_dfs_iterative <src.policies.helpers.operators.crossover.generalized_partition._dfs_iterative>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._dfs_iterative
    :summary:
    ```
* - {py:obj}`get_components <src.policies.helpers.operators.crossover.generalized_partition.get_components>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition.get_components
    :summary:
    ```
* - {py:obj}`generalized_partition_crossover <src.policies.helpers.operators.crossover.generalized_partition.generalized_partition_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition.generalized_partition_crossover
    :summary:
    ```
* - {py:obj}`_get_physical_edges <src.policies.helpers.operators.crossover.generalized_partition._get_physical_edges>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._get_physical_edges
    :summary:
    ```
* - {py:obj}`_get_components_from_uncommon_edges <src.policies.helpers.operators.crossover.generalized_partition._get_components_from_uncommon_edges>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._get_components_from_uncommon_edges
    :summary:
    ```
* - {py:obj}`_inherit_components <src.policies.helpers.operators.crossover.generalized_partition._inherit_components>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._inherit_components
    :summary:
    ```
* - {py:obj}`_greedy_pack_pool <src.policies.helpers.operators.crossover.generalized_partition._greedy_pack_pool>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._greedy_pack_pool
    :summary:
    ```
* - {py:obj}`_enforce_mandatory_nodes <src.policies.helpers.operators.crossover.generalized_partition._enforce_mandatory_nodes>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._enforce_mandatory_nodes
    :summary:
    ```
* - {py:obj}`route_profit_gpx_crossover <src.policies.helpers.operators.crossover.generalized_partition.route_profit_gpx_crossover>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition.route_profit_gpx_crossover
    :summary:
    ```
````

### API

````{py:function} _dfs_iterative(start: int, adj: typing.Dict[int, typing.List[int]], visited: typing.Set[int]) -> typing.List[int]
:canonical: src.policies.helpers.operators.crossover.generalized_partition._dfs_iterative

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._dfs_iterative
```
````

````{py:function} get_components(adj: typing.Dict[int, typing.List[int]], all_nodes: typing.Set[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.crossover.generalized_partition.get_components

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition.get_components
```
````

````{py:function} generalized_partition_crossover(p1: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, rng: typing.Optional[random.Random] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None, wastes: typing.Optional[typing.Dict[int, float]] = None, capacity: float = float('inf')) -> logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual
:canonical: src.policies.helpers.operators.crossover.generalized_partition.generalized_partition_crossover

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition.generalized_partition_crossover
```
````

````{py:function} _get_physical_edges(routes: typing.List[typing.List[int]]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.helpers.operators.crossover.generalized_partition._get_physical_edges

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._get_physical_edges
```
````

````{py:function} _get_components_from_uncommon_edges(p1_gt: typing.List[int], p2_gt: typing.List[int], uncommon_edges: typing.Set[typing.Tuple[int, int]]) -> typing.List[typing.List[int]]
:canonical: src.policies.helpers.operators.crossover.generalized_partition._get_components_from_uncommon_edges

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._get_components_from_uncommon_edges
```
````

````{py:function} _inherit_components(components: typing.List[typing.List[int]], p1: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, wastes: typing.Dict[int, float], capacity: float, rng: random.Random) -> typing.Tuple[typing.List[typing.List[int]], typing.Set[int]]
:canonical: src.policies.helpers.operators.crossover.generalized_partition._inherit_components

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._inherit_components
```
````

````{py:function} _greedy_pack_pool(pool: typing.List[int], child_routes: typing.List[typing.List[int]], child_nodes: typing.Set[int], loads: typing.List[float], dist_matrix: typing.Optional[numpy.ndarray], wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_set: typing.Set[int]) -> None
:canonical: src.policies.helpers.operators.crossover.generalized_partition._greedy_pack_pool

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._greedy_pack_pool
```
````

````{py:function} _enforce_mandatory_nodes(child_routes: typing.List[typing.List[int]], mandatory_nodes: typing.List[int], wastes: typing.Dict[int, float], capacity: float, loads: typing.List[float]) -> None
:canonical: src.policies.helpers.operators.crossover.generalized_partition._enforce_mandatory_nodes

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition._enforce_mandatory_nodes
```
````

````{py:function} route_profit_gpx_crossover(p1: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual, dist_matrix: typing.Optional[numpy.ndarray], wastes: typing.Dict[int, float], capacity: float, R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None, mandatory_nodes: typing.Optional[typing.List[int]] = None) -> logic.src.policies.route_construction.meta_heuristics.hybrid_genetic_search.individual.Individual
:canonical: src.policies.helpers.operators.crossover.generalized_partition.route_profit_gpx_crossover

```{autodoc2-docstring} src.policies.helpers.operators.crossover.generalized_partition.route_profit_gpx_crossover
```
````
