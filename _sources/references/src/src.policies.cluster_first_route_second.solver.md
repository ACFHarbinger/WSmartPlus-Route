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
````

### API

````{py:function} run_cf_rs(coords: pandas.DataFrame, must_go: typing.List[int], distance_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, n_vehicles: int, seed: int = 42, num_clusters: int = 0, time_limit: float = 60.0, assignment_method: str = 'greedy', route_optimizer: str = 'default', strict_fleet: bool = False, seed_criterion: str = 'distance', mip_objective: str = 'minimize_cost') -> typing.Tuple[typing.List[typing.List[int]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.cluster_first_route_second.solver.run_cf_rs

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver.run_cf_rs
```
````

````{py:function} fisher_jaikumar_clustering(coords: pandas.DataFrame, must_go: typing.List[int], k: int, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, distance_matrix: numpy.ndarray, time_limit: float = 60.0, assignment_method: str = 'greedy', strict_fleet: bool = True, seed_criterion: str = 'distance', mip_objective: str = 'minimize_cost') -> typing.List[typing.List[int]]
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

````{py:function} _select_initial_seeds(df_nodes: pandas.DataFrame, k: int, wastes: typing.Optional[typing.Dict[int, float]] = None, seed_criterion: str = 'distance') -> typing.List[int]
:canonical: src.policies.cluster_first_route_second.solver._select_initial_seeds

```{autodoc2-docstring} src.policies.cluster_first_route_second.solver._select_initial_seeds
```
````
