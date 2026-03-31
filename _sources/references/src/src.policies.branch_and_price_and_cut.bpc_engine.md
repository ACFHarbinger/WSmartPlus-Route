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

* - {py:obj}`_apply_branching_to_master <src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master
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
* - {py:obj}`_is_solution_integer <src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer
    :summary:
    ```
* - {py:obj}`_column_generation_loop <src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop
    :summary:
    ```
* - {py:obj}`run_custom_bpc <src.policies.branch_and_price_and_cut.bpc_engine.run_custom_bpc>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_custom_bpc
    :summary:
    ```
````

### API

````{py:function} _apply_branching_to_master(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, branching_constraints: typing.List) -> None
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._apply_branching_to_master
```
````

````{py:function} _separate_cuts(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, cut_engine: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine, max_cuts: int) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._separate_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._separate_cuts
```
````

````{py:function} _solve_pricing_step(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price.rcspp_dp.RCSPPSolver, branching_constraints: typing.Optional[typing.List] = None) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step
```
````

````{py:function} _is_solution_integer(route_values: typing.Dict[int, float], tol: float = 1e-06) -> bool
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._is_solution_integer
```
````

````{py:function} _column_generation_loop(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price.rcspp_dp.RCSPPSolver, cut_engine: src.policies.branch_and_price_and_cut.cutting_planes.CuttingPlaneEngine, branching_constraints: typing.Optional[typing.List], max_cg_iterations: int, max_cuts: int, time_limit: typing.Optional[float], start_time: float) -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._column_generation_loop
```
````

````{py:function} run_custom_bpc(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, expand_pool: bool = False, profit_aware_operators: bool = False, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine.run_custom_bpc

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_custom_bpc
```
````
