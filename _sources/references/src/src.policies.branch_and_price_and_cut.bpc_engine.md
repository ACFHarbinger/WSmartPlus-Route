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

* - {py:obj}`_separate_rcc <src.policies.branch_and_price_and_cut.bpc_engine._separate_rcc>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._separate_rcc
    :summary:
    ```
* - {py:obj}`_solve_pricing_step <src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step
    :summary:
    ```
* - {py:obj}`run_internal_bpc <src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc
    :summary:
    ```
````

### API

````{py:function} _separate_rcc(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, sep_engine: src.policies.branch_and_cut.separation.SeparationEngine, v_model: src.policies.branch_and_cut.vrpp_model.VRPPModel, max_cuts: int) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._separate_rcc

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._separate_rcc
```
````

````{py:function} _solve_pricing_step(master: src.policies.branch_and_price.master_problem.VRPPMasterProblem, pricing_solver: src.policies.branch_and_price.rcspp_dp.RCSPPSolver) -> int
:canonical: src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine._solve_pricing_step
```
````

````{py:function} run_internal_bpc(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.bpc_engine.run_internal_bpc
```
````
