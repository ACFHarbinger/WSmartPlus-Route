# {py:mod}`src.policies.branch_and_price_and_cut.bpc_engine`

```{py:module} src.policies.branch_and_price_and_cut.bpc_engine
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_reset_master_constraints <src.policies.branch_and_price_and_cut.bpc_engine._reset_master_constraints>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._reset_master_constraints
    :summary:
    ```
* - {py:obj}`_apply_route_level_branching_filters <src.policies.branch_and_price_and_cut.bpc_engine._apply_route_level_branching_filters>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_route_level_branching_filters
    :summary:
    ```
* - {py:obj}`_apply_branching_to_master <src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master
    :summary:
    ```
* - {py:obj}`_solve_farkas_pricing_step <src.policies.branch_and_price_and_cut.bpc_engine._solve_farkas_pricing_step>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_farkas_pricing_step
    :summary:
    ```
* - {py:obj}`_separate_cuts <src.policies.branch_and_price_and_cut.bpc_engine._separate_cuts>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._separate_cuts
    :summary:
    ```
* - {py:obj}`_solve_pricing_step <src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step
    :summary:
    ```
* - {py:obj}`_detect_cycles <src.policies.branch_and_price_and_cut.bpc_engine._detect_cycles>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._detect_cycles
    :summary:
    ```
* - {py:obj}`_is_solution_integer <src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer
    :summary:
    ```
* - {py:obj}`_perform_strong_branching <src.policies.branch_and_price_and_cut.bpc_engine._perform_strong_branching>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._perform_strong_branching
    :summary:
    ```
* - {py:obj}`_column_generation_loop <src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop
    :summary:
    ```
* - {py:obj}`run_bpc <src.policies.branch_and_price_and_cut.bpc_engine.run_bpc>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_bpc
    :summary:
    ```
* - {py:obj}`_apply_reduced_cost_edge_fixing <src.policies.branch_and_price_and_cut.bpc_engine._apply_reduced_cost_edge_fixing>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_reduced_cost_edge_fixing
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.branch_and_price_and_cut.bpc_engine.logger>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.branch_and_price_and_cut.bpc_engine.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.logger
```

````

````{py:exception} BPCPruningException()
:canonical: src.policies.branch_and_price_and_cut.bpc_engine.BPCPruningException

Bases: {py:obj}`Exception`

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.BPCPruningException
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.BPCPruningException.__init__
```

````

````{py:function} _reset_master_constraints(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem) -> None
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._reset_master_constraints

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._reset_master_constraints
```
````

````{py:function} _apply_route_level_branching_filters(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, bc: src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint) -> None
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._apply_route_level_branching_filters

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_route_level_branching_filters
```
````

````{py:function} _apply_branching_to_master(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, branching_constraints: typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint], branching_strategy: str = 'divergence') -> None
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master
```
````

````{py:function} _solve_farkas_pricing_step(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver, branching_constraints: typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint], farkas_duals: typing.Any, max_routes: int = 5) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._solve_farkas_pricing_step

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_farkas_pricing_step
```
````

````{py:function} _separate_cuts(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, cut_engine: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine, max_cuts: int, iteration: int = 0, node_depth: int = 0) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._separate_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._separate_cuts
```
````

````{py:function} _solve_pricing_step(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver, branching_constraints: typing.Optional[typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint]] = None, max_routes: int = 5, optimality_gap: float = 0.0001, rc_tolerance: float = 1e-05) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step
```
````

````{py:function} _detect_cycles(nodes: typing.List[int]) -> typing.List[typing.Tuple[int, ...]]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._detect_cycles

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._detect_cycles
```
````

````{py:function} _is_solution_integer(route_values: typing.Dict[int, float], tol: float = 1e-06) -> bool
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer
```
````

````{py:function} _perform_strong_branching(candidates: typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]], current_node: typing.Optional[src.policies.branch_and_price_and_cut.branching.BranchNode] = None) -> typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._perform_strong_branching

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._perform_strong_branching
```
````

````{py:function} _column_generation_loop(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver, cut_engine: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine, branching_constraints: typing.Optional[typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint]], max_cg_iterations: int, max_cuts: int, time_limit: typing.Optional[float], start_time: float, max_routes_per_pricing: int = 5, vehicle_limit: typing.Optional[int] = None, optimality_gap: float = 0.0001, early_termination_gap: float = 0.001, parent_basis: typing.Optional[typing.Any] = None, incumbent_value: float = -float('inf'), node_depth: int = 0) -> typing.Tuple[float, typing.Dict[int, float], typing.Optional[typing.Any]]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop
```
````

````{py:function} run_bpc(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[typing.Union[src.policies.branch_and_price_and_cut.params.BPCParams, typing.Dict[str, typing.Any]]] = None, must_go_indices: typing.Optional[typing.Set[int]] = None, vehicle_limit: typing.Optional[int] = None, env: typing.Optional[typing.Any] = None, node_coords: typing.Optional[numpy.ndarray] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine.run_bpc

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_bpc
```
````

````{py:function} _apply_reduced_cost_edge_fixing(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price_and_cut.rcspp_dp.RCSPPSolver, z_ub: float, z_lb: float) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._apply_reduced_cost_edge_fixing

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_reduced_cost_edge_fixing
```
````
