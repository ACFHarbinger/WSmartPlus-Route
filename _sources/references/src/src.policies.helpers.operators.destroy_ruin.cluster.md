# {py:mod}`src.policies.helpers.operators.destroy_ruin.cluster`

```{py:module} src.policies.helpers.operators.destroy_ruin.cluster
```

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_kruskal_two_clusters <src.policies.helpers.operators.destroy_ruin.cluster._kruskal_two_clusters>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster._kruskal_two_clusters
    :summary:
    ```
* - {py:obj}`_find_cross_route_neighbor <src.policies.helpers.operators.destroy_ruin.cluster._find_cross_route_neighbor>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster._find_cross_route_neighbor
    :summary:
    ```
* - {py:obj}`cluster_removal <src.policies.helpers.operators.destroy_ruin.cluster.cluster_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster.cluster_removal
    :summary:
    ```
* - {py:obj}`cluster_profit_removal <src.policies.helpers.operators.destroy_ruin.cluster.cluster_profit_removal>`
  - ```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster.cluster_profit_removal
    :summary:
    ```
````

### API

````{py:function} _kruskal_two_clusters(route: typing.List[int], dist_matrix: numpy.ndarray, d_max: float) -> typing.Tuple[typing.List[int], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.cluster._kruskal_two_clusters

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster._kruskal_two_clusters
```
````

````{py:function} _find_cross_route_neighbor(pivot: int, pivot_route_idx: int, routes: typing.List[typing.List[int]], removed_set: typing.Set[int], dist_matrix: numpy.ndarray) -> typing.Optional[typing.Tuple[int, int]]
:canonical: src.policies.helpers.operators.destroy_ruin.cluster._find_cross_route_neighbor

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster._find_cross_route_neighbor
```
````

````{py:function} cluster_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, nodes: typing.List[int], rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.cluster.cluster_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster.cluster_removal
```
````

````{py:function} cluster_profit_removal(routes: typing.List[typing.List[int]], n_remove: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], R: float = 1.0, C: float = 1.0, rng: typing.Optional[random.Random] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int]]
:canonical: src.policies.helpers.operators.destroy_ruin.cluster.cluster_profit_removal

```{autodoc2-docstring} src.policies.helpers.operators.destroy_ruin.cluster.cluster_profit_removal
```
````
