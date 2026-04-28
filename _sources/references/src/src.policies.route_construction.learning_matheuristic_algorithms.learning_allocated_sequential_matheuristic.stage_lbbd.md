# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_LBBDMaster <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_route_cost_from_nodes <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_cost_from_nodes>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_cost_from_nodes
    :summary:
    ```
* - {py:obj}`_route_revenue_from_nodes <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_revenue_from_nodes>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_revenue_from_nodes
    :summary:
    ```
* - {py:obj}`_make_vrpp_route <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._make_vrpp_route>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._make_vrpp_route
    :summary:
    ```
* - {py:obj}`_solve_sub_greedy <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_greedy>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_greedy
    :summary:
    ```
* - {py:obj}`_solve_sub_alns <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_alns>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_alns
    :summary:
    ```
* - {py:obj}`_solve_sub_bpc <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_bpc>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_bpc
    :summary:
    ```
* - {py:obj}`run_lbbd_stage <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.run_lbbd_stage>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.run_lbbd_stage
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.logger
```

````

````{py:function} _route_cost_from_nodes(nodes: typing.List[int], dist: numpy.ndarray, C: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_cost_from_nodes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_cost_from_nodes
```
````

````{py:function} _route_revenue_from_nodes(nodes: typing.List[int], wastes: typing.Dict[int, float], R: float) -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_revenue_from_nodes

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._route_revenue_from_nodes
```
````

````{py:function} _make_vrpp_route(nodes: typing.List[int], dist: numpy.ndarray, wastes: typing.Dict[int, float], R: float, C: float, source: str) -> logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._make_vrpp_route

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._make_vrpp_route
```
````

````{py:function} _solve_sub_greedy(selected: typing.Set[int], dist: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_greedy

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_greedy
```
````

````{py:function} _solve_sub_alns(selected: typing.Set[int], dist: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], time_limit: float, seed: typing.Optional[int]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_alns

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_alns
```
````

````{py:function} _solve_sub_bpc(selected: typing.Set[int], dist: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], time_limit: float, seed: typing.Optional[int], vehicle_limit: typing.Optional[int], env: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_bpc

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._solve_sub_bpc
```
````

`````{py:class} _LBBDMaster(n_nodes: int, wastes: typing.Dict[int, float], R: float, mandatory: typing.Set[int], min_cover_ratio: float, vehicle_limit: typing.Optional[int], env: typing.Any, seed: int)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.__init__
```

````{py:method} solve(time_limit: float) -> typing.Optional[typing.Set[int]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.solve

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.solve
```

````

````{py:method} get_lp_bound() -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.get_lp_bound

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.get_lp_bound
```

````

````{py:method} add_nogood_cut(Y: typing.Set[int]) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_nogood_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_nogood_cut
```

````

````{py:method} add_optimality_cut(Y: typing.Set[int], sub_cost: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_optimality_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_optimality_cut
```

````

````{py:method} add_pareto_cut(Y: typing.Set[int], sub_cost: float, primal_bound: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_pareto_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_pareto_cut
```

````

````{py:method} add_combinatorial_cut(Y: typing.Set[int], sub_cost: float) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_combinatorial_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.add_combinatorial_cut
```

````

````{py:property} n_cuts
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.n_cuts
:type: int

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd._LBBDMaster.n_cuts
```

````

`````

````{py:function} run_lbbd_stage(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], n_vehicles: int, time_limit: float, max_iterations: int, sub_solver: str, cut_families: typing.List[str], pareto_eps: float, min_cover_ratio: float, master_time_frac: float, sub_time_frac: float, pool: typing.Optional[logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.RoutePool], seed: int, env: typing.Any, incumbent: float = 0.0) -> typing.Tuple[typing.List[logic.src.policies.route_construction.matheuristics.exact_guided_heuristic.route_pool.VRPPRoute], float, float]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.run_lbbd_stage

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.learning_allocated_sequential_matheuristic.stage_lbbd.run_lbbd_stage
```
````
