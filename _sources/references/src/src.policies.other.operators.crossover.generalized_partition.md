# {py:mod}`src.policies.other.operators.crossover.generalized_partition`

```{py:module} src.policies.other.operators.crossover.generalized_partition
```

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`get_edges <src.policies.other.operators.crossover.generalized_partition.get_edges>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.get_edges
    :summary:
    ```
* - {py:obj}`get_components <src.policies.other.operators.crossover.generalized_partition.get_components>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.get_components
    :summary:
    ```
* - {py:obj}`generalized_partition_crossover <src.policies.other.operators.crossover.generalized_partition.generalized_partition_crossover>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.generalized_partition_crossover
    :summary:
    ```
* - {py:obj}`_get_physical_edges <src.policies.other.operators.crossover.generalized_partition._get_physical_edges>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition._get_physical_edges
    :summary:
    ```
* - {py:obj}`_get_physical_components <src.policies.other.operators.crossover.generalized_partition._get_physical_components>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition._get_physical_components
    :summary:
    ```
* - {py:obj}`route_profit_gpx_crossover <src.policies.other.operators.crossover.generalized_partition.route_profit_gpx_crossover>`
  - ```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.route_profit_gpx_crossover
    :summary:
    ```
````

### API

````{py:function} get_edges(tour: typing.List[int]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.other.operators.crossover.generalized_partition.get_edges

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.get_edges
```
````

````{py:function} get_components(adj: typing.Dict[int, typing.List[int]], all_nodes: typing.Set[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.crossover.generalized_partition.get_components

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.get_components
```
````

````{py:function} generalized_partition_crossover(p1: logic.src.policies.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.hybrid_genetic_search.individual.Individual, rng: typing.Optional[random.Random] = None) -> logic.src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.other.operators.crossover.generalized_partition.generalized_partition_crossover

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.generalized_partition_crossover
```
````

````{py:function} _get_physical_edges(routes: typing.List[typing.List[int]]) -> typing.Set[typing.Tuple[int, int]]
:canonical: src.policies.other.operators.crossover.generalized_partition._get_physical_edges

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition._get_physical_edges
```
````

````{py:function} _get_physical_components(adj: typing.Dict[int, typing.List[int]], all_nodes: typing.Set[int]) -> typing.List[typing.List[int]]
:canonical: src.policies.other.operators.crossover.generalized_partition._get_physical_components

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition._get_physical_components
```
````

````{py:function} route_profit_gpx_crossover(p1: logic.src.policies.hybrid_genetic_search.individual.Individual, p2: logic.src.policies.hybrid_genetic_search.individual.Individual, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None) -> logic.src.policies.hybrid_genetic_search.individual.Individual
:canonical: src.policies.other.operators.crossover.generalized_partition.route_profit_gpx_crossover

```{autodoc2-docstring} src.policies.other.operators.crossover.generalized_partition.route_profit_gpx_crossover
```
````
