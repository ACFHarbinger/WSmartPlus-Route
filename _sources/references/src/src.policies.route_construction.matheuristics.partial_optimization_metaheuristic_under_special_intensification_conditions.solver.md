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

* - {py:obj}`_compute_d_assign <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_assign>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_assign
    :summary:
    ```
* - {py:obj}`_compute_d_prox <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_prox>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_prox
    :summary:
    ```
* - {py:obj}`_pmedian_alternating <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._pmedian_alternating>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._pmedian_alternating
    :summary:
    ```
* - {py:obj}`_build_proximity_network <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._build_proximity_network>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._build_proximity_network
    :summary:
    ```
* - {py:obj}`_initialize_pmedian <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._initialize_pmedian>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._initialize_pmedian
    :summary:
    ```
* - {py:obj}`_route_centroid <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._route_centroid>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._route_centroid
    :summary:
    ```
* - {py:obj}`_assemble_subproblem <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._assemble_subproblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._assemble_subproblem
    :summary:
    ```
* - {py:obj}`run_popmusic <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic
    :summary:
    ```
* - {py:obj}`_evaluate_slots <src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._evaluate_slots>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._evaluate_slots
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
````

### API

````{py:function} _compute_d_assign(node_coords: numpy.ndarray, center_coords: numpy.ndarray, revenues: numpy.ndarray, d_max: float, rev_max: float, eps: float = 1e-08) -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_assign

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_assign
```
````

````{py:function} _compute_d_prox(center_coords: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_prox

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._compute_d_prox
```
````

````{py:function} _pmedian_alternating(node_coords: numpy.ndarray, revenues: numpy.ndarray, p: int, rng: random.Random, max_iter: int = 30, profit_aware: bool = True) -> typing.Tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._pmedian_alternating

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._pmedian_alternating
```
````

````{py:function} _build_proximity_network(center_coords: numpy.ndarray, assignments: numpy.ndarray, d_assign_matrix: numpy.ndarray) -> typing.Dict[int, typing.List[int]]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._build_proximity_network

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._build_proximity_network
```
````

````{py:function} _initialize_pmedian(coords_df: pandas.DataFrame, wastes_dict: typing.Dict[int, float], capacity: float, R: float, C: float, n_vehicles: int, subproblem_size: int, seed: int, profit_aware: bool, distance_matrix: numpy.ndarray, mandatory: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], numpy.ndarray, typing.Dict[int, typing.List[int]]]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._initialize_pmedian

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._initialize_pmedian
```
````

````{py:function} _route_centroid(nodes: typing.List[int], coord_array: numpy.ndarray) -> numpy.ndarray
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._route_centroid

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._route_centroid
```
````

````{py:function} _assemble_subproblem(seed_slot: int, state: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState, r: int) -> typing.List[int]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._assemble_subproblem

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._assemble_subproblem
```
````

````{py:function} run_popmusic(coords: pandas.DataFrame, mandatory: typing.List[int], distance_matrix: numpy.ndarray, n_vehicles: int, subproblem_size: int = 3, max_iterations: typing.Optional[int] = None, base_solver: str = 'fast_tsp', base_solver_config: typing.Optional[typing.Any] = None, cluster_solver: str = 'fast_tsp', cluster_solver_config: typing.Optional[typing.Any] = None, initial_solver: str = 'pmedian', seed: int = 42, wastes: typing.Optional[typing.Dict[int, float]] = None, capacity: float = 1000000000.0, R: float = 1.0, C: float = 0.0, vrpp: bool = True, profit_aware_operators: bool = False, k_prox: int = 10, seed_strategy: str = 'lifo') -> typing.Tuple[typing.List[typing.List[int]], float, float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver.run_popmusic
```
````

````{py:function} _evaluate_slots(slots: typing.List[int], state: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.state.POPMUSICState, distance_matrix: numpy.ndarray, wastes_dict: typing.Dict[int, float], R: float, C: float) -> float
:canonical: src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._evaluate_slots

```{autodoc2-docstring} src.policies.route_construction.matheuristics.partial_optimization_metaheuristic_under_special_intensification_conditions.solver._evaluate_slots
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
