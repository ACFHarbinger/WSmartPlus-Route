# {py:mod}`src.policies.cluster_first_route_second.solver`

```{py:module} src.policies.cluster_first_route_second.solver
```

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_cf_rs <src.policies.cluster_first_route_second.solver.run_cf_rs>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.run_cf_rs
    :summary:
    ```
* - {py:obj}`fisher_jaikumar_clustering <src.policies.cluster_first_route_second.solver.fisher_jaikumar_clustering>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.fisher_jaikumar_clustering
    :summary:
    ```
* - {py:obj}`_get_depot_coords <src.policies.cluster_first_route_second.solver._get_depot_coords>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._get_depot_coords
    :summary:
    ```
* - {py:obj}`_compute_node_features <src.policies.cluster_first_route_second.solver._compute_node_features>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._compute_node_features
    :summary:
    ```
* - {py:obj}`_select_initial_seeds <src.policies.cluster_first_route_second.solver._select_initial_seeds>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._select_initial_seeds
    :summary:
    ```
* - {py:obj}`_perform_gap_assignment <src.policies.cluster_first_route_second.solver._perform_gap_assignment>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._perform_gap_assignment
    :summary:
    ```
* - {py:obj}`_find_best_seed <src.policies.cluster_first_route_second.solver._find_best_seed>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._find_best_seed
    :summary:
    ```
* - {py:obj}`_bounded_partition <src.policies.cluster_first_route_second.solver._bounded_partition>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._bounded_partition
    :summary:
    ```
* - {py:obj}`_unbounded_partition <src.policies.cluster_first_route_second.solver._unbounded_partition>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._unbounded_partition
    :summary:
    ```
````

### API

````{py:function} run_cf_rs(coords: pandas.DataFrame, must_go: typing.List[int], distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, n_vehicles: int, seed: int = 42, num_clusters: int = 0, time_limit: float = 60.0) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.cluster_first_route_second.solver.run_cf_rs

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.run_cf_rs
```
````

````{py:function} fisher_jaikumar_clustering(coords: pandas.DataFrame, must_go: typing.List[int], k: int, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, distance_matrix: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.cluster_first_route_second.solver.fisher_jaikumar_clustering

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.fisher_jaikumar_clustering
```
````

````{py:function} _get_depot_coords(coords: typing.Any) -> typing.Tuple[float, float]
:canonical: src.policies.cluster_first_route_second.solver._get_depot_coords

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._get_depot_coords
```
````

````{py:function} _compute_node_features(coords: typing.Any, must_go: typing.List[int], depot_lat: float, depot_lng: float, distance_matrix: numpy.ndarray) -> pandas.DataFrame
:canonical: src.policies.cluster_first_route_second.solver._compute_node_features

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._compute_node_features
```
````

````{py:function} _select_initial_seeds(df_nodes: pandas.DataFrame, k: int) -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.solver._select_initial_seeds

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._select_initial_seeds
```
````

````{py:function} _perform_gap_assignment(seeds: typing.List[int], must_go: typing.List[int], wastes: typing.Dict[int, float], capacity: float, distance_matrix: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.cluster_first_route_second.solver._perform_gap_assignment

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._perform_gap_assignment
```
````

````{py:function} _find_best_seed(node: int, seeds: typing.List[int], loads: typing.List[float], wastes: typing.Dict[int, float], capacity: float, distance_matrix: numpy.ndarray) -> int
:canonical: src.policies.cluster_first_route_second.solver._find_best_seed

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._find_best_seed
```
````

````{py:function} _bounded_partition(sorted_indices: typing.List[int], k: int, wastes: typing.Dict[int, float], capacity: float, mandatory_set: set) -> typing.List[typing.List[int]]
:canonical: src.policies.cluster_first_route_second.solver._bounded_partition

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._bounded_partition
```
````

````{py:function} _unbounded_partition(sorted_indices: typing.List[int], wastes: typing.Dict[int, float], capacity: float, mandatory_set: set) -> typing.List[typing.List[int]]
:canonical: src.policies.cluster_first_route_second.solver._unbounded_partition

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._unbounded_partition
```
````
