# {py:mod}`src.policies.lin_kernighan_helsgaun_three.lkh3`

```{py:module} src.policies.lin_kernighan_helsgaun_three.lkh3
```

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_improve_tour <src.policies.lin_kernighan_helsgaun_three.lkh3._improve_tour>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3._improve_tour
    :summary:
    ```
* - {py:obj}`solve_lkh3 <src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3
    :summary:
    ```
* - {py:obj}`solve_lkh3_with_alns <src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3_with_alns>`
  - ```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3_with_alns
    :summary:
    ```
````

### API

````{py:function} _improve_tour(curr_tour: typing.List[int], curr_pen: float, curr_cost: float, candidates: typing.Dict[int, typing.List[int]], distance_matrix: numpy.ndarray, waste: typing.Optional[numpy.ndarray], capacity: typing.Optional[float], rng: random.Random, dont_look_bits: typing.Optional[numpy.ndarray] = None, max_k_opt: int = 5, n_original: typing.Optional[int] = None, dynamic_topology_discovery: bool = False) -> typing.Tuple[typing.List[int], float, float, bool, typing.Optional[numpy.ndarray]]
:canonical: src.policies.lin_kernighan_helsgaun_three.lkh3._improve_tour

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3._improve_tour
```
````

````{py:function} solve_lkh3(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, waste: typing.Optional[numpy.ndarray] = None, capacity: float = 100.0, revenue: float = 1.0, cost_unit: float = 1.0, mandatory_nodes: typing.Optional[typing.List[int]] = None, coords: typing.Optional[numpy.ndarray] = None, max_trials: int = 100, popmusic_subpath_size: int = 50, popmusic_trials: int = 50, popmusic_max_candidates: int = 5, max_k_opt: int = 5, use_ip_merging: bool = True, max_pool_size: int = 5, subgradient_iterations: int = 50, profit_aware_operators: bool = False, alns_iterations: int = 10, plateau_limit: int = 10, deep_plateau_limit: int = 30, perturb_operator_weights: typing.Optional[typing.List[float]] = None, n_vehicles: int = 0, n_original: int = 0, candidate_set: typing.Optional[typing.Dict[int, typing.List[int]]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, rng: typing.Optional[random.Random] = None, seed: int = 42, dynamic_topology_discovery: bool = False, native_prize_collecting: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3
```
````

````{py:function} solve_lkh3_with_alns(distance_matrix: numpy.ndarray, initial_tour: typing.Optional[typing.List[int]] = None, waste: typing.Optional[numpy.ndarray] = None, capacity: float = 100.0, revenue: float = 1.0, cost_unit: float = 1.0, mandatory_nodes: typing.Optional[typing.List[int]] = None, coords: typing.Optional[numpy.ndarray] = None, max_trials: int = 100, popmusic_subpath_size: int = 50, popmusic_trials: int = 50, popmusic_max_candidates: int = 5, max_k_opt: int = 5, use_ip_merging: bool = True, max_pool_size: int = 5, subgradient_iterations: int = 50, profit_aware_operators: bool = False, alns_iterations: int = 100, plateau_limit: int = 10, deep_plateau_limit: int = 30, perturb_operator_weights: typing.Optional[typing.List[float]] = None, n_vehicles: int = 0, n_original: int = 0, candidate_set: typing.Optional[typing.Dict[int, typing.List[int]]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, np_rng: typing.Optional[numpy.random.Generator] = None, rng: typing.Optional[random.Random] = None, seed: int = 42, dynamic_topology_discovery: bool = False, native_prize_collecting: bool = False) -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3_with_alns

```{autodoc2-docstring} src.policies.lin_kernighan_helsgaun_three.lkh3.solve_lkh3_with_alns
```
````
