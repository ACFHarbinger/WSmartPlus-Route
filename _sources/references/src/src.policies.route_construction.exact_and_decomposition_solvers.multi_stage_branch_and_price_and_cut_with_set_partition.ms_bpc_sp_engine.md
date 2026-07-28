# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_reset_master_constraints <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._reset_master_constraints>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._reset_master_constraints
    :summary:
    ```
* - {py:obj}`_apply_route_level_branching_filters <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_route_level_branching_filters>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_route_level_branching_filters
    :summary:
    ```
* - {py:obj}`_apply_branching_to_master <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_branching_to_master>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_branching_to_master
    :summary:
    ```
* - {py:obj}`_solve_farkas_pricing_step <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_farkas_pricing_step>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_farkas_pricing_step
    :summary:
    ```
* - {py:obj}`_separate_cuts <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._separate_cuts>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._separate_cuts
    :summary:
    ```
* - {py:obj}`_solve_pricing_step <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_pricing_step>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_pricing_step
    :summary:
    ```
* - {py:obj}`_detect_cycles <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._detect_cycles>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._detect_cycles
    :summary:
    ```
* - {py:obj}`_is_solution_integer <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._is_solution_integer>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._is_solution_integer
    :summary:
    ```
* - {py:obj}`_perform_strong_branching <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._perform_strong_branching>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._perform_strong_branching
    :summary:
    ```
* - {py:obj}`_compute_lr_bound_at_node <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._compute_lr_bound_at_node>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._compute_lr_bound_at_node
    :summary:
    ```
* - {py:obj}`_extract_forced_sets_from_constraints <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._extract_forced_sets_from_constraints>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._extract_forced_sets_from_constraints
    :summary:
    ```
* - {py:obj}`_column_generation_loop <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._column_generation_loop>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._column_generation_loop
    :summary:
    ```
* - {py:obj}`_select_nodes_knapsack <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._select_nodes_knapsack>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._select_nodes_knapsack
    :summary:
    ```
* - {py:obj}`run_ms_bpc_sp <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.run_ms_bpc_sp>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.run_ms_bpc_sp
    :summary:
    ```
* - {py:obj}`_apply_reduced_cost_edge_fixing <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_reduced_cost_edge_fixing>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_reduced_cost_edge_fixing
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.logger
    :summary:
    ```
* - {py:obj}`_FARKAS_TOL <src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._FARKAS_TOL>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._FARKAS_TOL
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.logger
```

````

````{py:data} _FARKAS_TOL
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._FARKAS_TOL
:type: float
:value: >
   1e-06

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._FARKAS_TOL
```

````

````{py:exception} MSBPCSPPruningException()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.MSBPCSPPruningException

Bases: {py:obj}`Exception`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.MSBPCSPPruningException
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.MSBPCSPPruningException.__init__
```

````

````{py:function} _reset_master_constraints(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._reset_master_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._reset_master_constraints
```
````

````{py:function} _apply_route_level_branching_filters(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, bc: logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_route_level_branching_filters

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_route_level_branching_filters
```
````

````{py:function} _apply_branching_to_master(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, branching_constraints: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint], branching_strategy: str = 'divergence') -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_branching_to_master

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_branching_to_master
```
````

````{py:function} _solve_farkas_pricing_step(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver, branching_constraints: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint], farkas_duals: typing.Any, max_routes: int = 5, timeout: float = 5.0) -> typing.Tuple[int, bool]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_farkas_pricing_step

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_farkas_pricing_step
```
````

````{py:function} _separate_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, cut_engine: logic.src.policies.helpers.solvers_and_matheuristics.CuttingPlaneEngine, max_cuts: int, iteration: int = 0, node_depth: int = 0, cut_orthogonality_threshold: float = 0.8) -> int
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._separate_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._separate_cuts
```
````

````{py:function} _solve_pricing_step(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint]] = None, max_routes: int = 5, optimality_gap: float = 0.0001, rc_tolerance: float = 1e-05, timeout: typing.Optional[float] = None) -> typing.Tuple[int, bool]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_pricing_step

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._solve_pricing_step
```
````

````{py:function} _detect_cycles(nodes: typing.List[int]) -> typing.List[typing.Tuple[int, ...]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._detect_cycles

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._detect_cycles
```
````

````{py:function} _is_solution_integer(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.Route], route_values: typing.Dict[int, float], tol: float = 1e-06) -> bool
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._is_solution_integer

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._is_solution_integer
```
````

````{py:function} _perform_strong_branching(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, candidates: typing.List[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]], current_node: typing.Optional[logic.src.policies.helpers.solvers_and_matheuristics.BranchNode] = None, strong_branching_size: int = 5) -> typing.Optional[typing.Tuple[int, typing.List[typing.Tuple[int, int]], typing.List[typing.Tuple[int, int]], float]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._perform_strong_branching

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._perform_strong_branching
```
````

````{py:function} _compute_lr_bound_at_node(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], forced_out: typing.Set[int], params: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams, time_budget: float, env: typing.Optional[typing.Any], recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder]) -> typing.Tuple[float, float, typing.Set[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._compute_lr_bound_at_node

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._compute_lr_bound_at_node
```
````

````{py:function} _extract_forced_sets_from_constraints(branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint]]) -> typing.Tuple[typing.Set[int], typing.Set[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._extract_forced_sets_from_constraints

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._extract_forced_sets_from_constraints
```
````

````{py:function} _column_generation_loop(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver, cut_engine: logic.src.policies.helpers.solvers_and_matheuristics.CuttingPlaneEngine, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.AnyBranchingConstraint]], max_cg_iterations: int, max_cuts: int, time_limit: float, start_time: float, max_routes_per_pricing: int = 5, vehicle_limit: typing.Optional[int] = None, optimality_gap: float = 0.0001, early_termination_gap: float = 0.001, parent_basis: typing.Optional[typing.Any] = None, incumbent_value: float = -float('inf'), node_depth: int = 0, rc_tolerance: float = 1e-05, cut_orthogonality_threshold: float = 0.8, exact_mode: bool = False, cg_at_root_only: bool = False, branching_strategy: str = 'divergence', rcspp_timeout: float = 30.0) -> typing.Tuple[float, typing.Dict[int, float], typing.Optional[typing.Any], bool]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._column_generation_loop

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._column_generation_loop
```
````

````{py:function} _select_nodes_knapsack(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], n_nodes: int, vehicle_limit: typing.Optional[int] = None, target_reduction: float = 0.6, time_limit: float = 10.0, env: typing.Optional[typing.Any] = None) -> typing.Set[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._select_nodes_knapsack

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._select_nodes_knapsack
```
````

````{py:function} run_ms_bpc_sp(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[typing.Union[src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.params.MSBPCSPParams, typing.Dict[str, typing.Any]]] = None, mandatory_indices: typing.Optional[typing.Set[int]] = None, vehicle_limit: typing.Optional[int] = None, env: typing.Optional[typing.Any] = None, node_coords: typing.Optional[numpy.ndarray] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.run_ms_bpc_sp

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine.run_ms_bpc_sp
```
````

````{py:function} _apply_reduced_cost_edge_fixing(master: logic.src.policies.helpers.solvers_and_matheuristics.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.RCSPPSolver, z_ub: float, z_lb: float) -> int
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_reduced_cost_edge_fixing

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.multi_stage_branch_and_price_and_cut_with_set_partition.ms_bpc_sp_engine._apply_reduced_cost_edge_fixing
```
````
