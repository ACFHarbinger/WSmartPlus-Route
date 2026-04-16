# {py:mod}`src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver`

```{py:module} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_popmusic <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic
    :summary:
    ```
* - {py:obj}`_optimize_subproblem <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_subproblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_subproblem
    :summary:
    ```
* - {py:obj}`_optimize_with_fast_tsp <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_fast_tsp>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_fast_tsp
    :summary:
    ```
* - {py:obj}`_optimize_with_hgs <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_hgs>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_hgs
    :summary:
    ```
* - {py:obj}`_optimize_with_alns <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_alns>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_alns
    :summary:
    ```
* - {py:obj}`find_route_neighbors <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.find_route_neighbors>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.find_route_neighbors
    :summary:
    ```
* - {py:obj}`split_tour <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.split_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.split_tour
    :summary:
    ```
````

### API

````{py:function} run_popmusic(coords: pandas.DataFrame, mandatory: typing.List[int], distance_matrix: numpy.ndarray, n_vehicles: int, subproblem_size: int = 3, max_iterations: typing.Optional[int] = None, base_solver: str = 'fast_tsp', base_solver_config: typing.Optional[typing.Any] = None, cluster_solver: str = 'fast_tsp', cluster_solver_config: typing.Optional[typing.Any] = None, initial_solver: str = 'nearest_neighbor', seed: int = 42, wastes: typing.Optional[typing.Dict[int, float]] = None, capacity: float = 1000000000.0, R: float = 1.0, C: float = 0.0, vrpp: bool = True, profit_aware_operators: bool = False, k_prox: int = 10, seed_strategy: str = 'lifo') -> typing.Tuple[typing.List[typing.List[int]], float, float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic
```
````

````{py:function} _optimize_subproblem(base_solver: typing.Optional[str], base_solver_config: typing.Optional[typing.Any], subproblem_nodes: typing.List[int], distance_matrix: numpy.ndarray, wastes_dict: typing.Dict[int, float], capacity: float, R: float, C: float, neighborhood_indices: typing.List[int], mandatory: typing.List[int], seed: int, vrpp: bool = True, profit_aware_operators: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_subproblem

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_subproblem
```
````

````{py:function} _optimize_with_fast_tsp(subproblem_nodes: typing.List[int], distance_matrix: numpy.ndarray, wastes_dict: typing.Dict[int, float], capacity: float, R: float, C: float, config: typing.Optional[typing.Any], time_limit: float, seed: int, vrpp: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_fast_tsp

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_fast_tsp
```
````

````{py:function} _optimize_with_hgs(distance_matrix: numpy.ndarray, wastes_dict: typing.Dict[int, float], capacity: float, R: float, C: float, neighborhood_indices: typing.List[int], mandatory: typing.List[int], config: typing.Optional[typing.Any], time_limit: float, seed: int, vrpp: bool = True, profit_aware_operators: bool = False, subproblem_nodes: typing.Optional[typing.List[int]] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_hgs

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_hgs
```
````

````{py:function} _optimize_with_alns(distance_matrix: numpy.ndarray, wastes_dict: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.List[int], config: typing.Optional[typing.Any], time_limit: float, seed: int, vrpp: bool = True, profit_aware_operators: bool = False, subproblem_nodes: typing.Optional[typing.List[int]] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_alns

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._optimize_with_alns
```
````

````{py:function} find_route_neighbors(seed_idx: int, centroids: typing.List[numpy.ndarray], k: int, kdtree: typing.Optional[scipy.spatial.KDTree] = None, k_prox: int = 0) -> typing.List[int]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.find_route_neighbors

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.find_route_neighbors
```
````

````{py:function} split_tour(tour: typing.List[int], k: int, distance_matrix: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.split_tour

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.split_tour
```
````
