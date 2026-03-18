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
* - {py:obj}`angular_clustering <src.policies.cluster_first_route_second.solver.angular_clustering>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.angular_clustering
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
* - {py:obj}`calculate_tour_cost <src.policies.cluster_first_route_second.solver.calculate_tour_cost>`
  - ```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.calculate_tour_cost
    :summary:
    ```
````

### API

````{py:function} run_cf_rs(coords: pandas.DataFrame, must_go: typing.List[int], distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, n_vehicles: int, seed: int = 42, num_clusters: int = 0, time_limit: float = 60.0) -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.cluster_first_route_second.solver.run_cf_rs

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.run_cf_rs
```
````

````{py:function} angular_clustering(coords: pandas.DataFrame, must_go: typing.List[int], k: int, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, distance_matrix: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.cluster_first_route_second.solver.angular_clustering

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.angular_clustering
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

````{py:function} calculate_tour_cost(distance_matrix: numpy.ndarray, tour: typing.List[int]) -> float
:canonical: src.policies.cluster_first_route_second.solver.calculate_tour_cost

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.calculate_tour_cost
```
````
