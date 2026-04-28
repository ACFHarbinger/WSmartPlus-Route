# {py:mod}`src.policies.helpers.solvers_and_matheuristics.pricing.smoothing`

```{py:module} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`detect_cycles <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.detect_cycles>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.detect_cycles
    :summary:
    ```
* - {py:obj}`is_solution_integer <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.is_solution_integer>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.is_solution_integer
    :summary:
    ```
* - {py:obj}`separate_cuts <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.separate_cuts>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.separate_cuts
    :summary:
    ```
* - {py:obj}`solve_farkas_pricing_step <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_farkas_pricing_step>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_farkas_pricing_step
    :summary:
    ```
* - {py:obj}`solve_pricing_step <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_pricing_step>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_pricing_step
    :summary:
    ```
* - {py:obj}`dssr_pricing_wrapper <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.dssr_pricing_wrapper>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.dssr_pricing_wrapper
    :summary:
    ```
* - {py:obj}`reduced_cost_arc_fixing <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.reduced_cost_arc_fixing>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.reduced_cost_arc_fixing
    :summary:
    ```
* - {py:obj}`apply_reduced_cost_edge_fixing <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.apply_reduced_cost_edge_fixing>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.apply_reduced_cost_edge_fixing
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.logger
```

````

````{py:function} detect_cycles(nodes: typing.List[int]) -> typing.List[typing.Tuple[int, ...]]
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.detect_cycles

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.detect_cycles
```
````

````{py:function} is_solution_integer(routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.Route], route_values: typing.Dict[int, float], tol: float = 1e-06) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.is_solution_integer

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.is_solution_integer
```
````

````{py:function} separate_cuts(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, cut_engine: logic.src.policies.helpers.solvers_and_matheuristics.search.cutting_planes.CuttingPlaneEngine, max_cuts: int, iteration: int = 0, node_depth: int = 0, cut_orthogonality_threshold: float = 0.8) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.separate_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.separate_cuts
```
````

````{py:function} solve_farkas_pricing_step(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver.RCSPPSolver, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.AnyBranchingConstraint]] = None, max_routes: int = 5, timeout: typing.Optional[float] = None) -> typing.Tuple[int, bool]
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_farkas_pricing_step

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_farkas_pricing_step
```
````

````{py:function} solve_pricing_step(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, pricing_solver: logic.src.policies.helpers.solvers_and_matheuristics.pricing.solver.RCSPPSolver, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.AnyBranchingConstraint]] = None, max_routes: int = 5, optimality_gap: float = 0.0001, rc_tolerance: float = 1e-05, timeout: typing.Optional[float] = None, use_dssr: bool = False, dssr_max_iters: int = 8) -> typing.Tuple[int, bool]
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_pricing_step

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.solve_pricing_step
```
````

````{py:function} dssr_pricing_wrapper(pricing_solver: typing.Any, node_duals: typing.Dict[int, typing.Any], max_routes: int, forced_nodes: typing.Optional[typing.Set[int]] = None, rf_conflicts: typing.Optional[typing.Dict] = None, forbidden_arcs: typing.Optional[typing.FrozenSet] = None, required_successors: typing.Optional[typing.Dict] = None, required_predecessors: typing.Optional[typing.Dict] = None, timeout: typing.Optional[float] = None, max_dssr_iters: int = 8) -> typing.List[typing.Any]
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.dssr_pricing_wrapper

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.dssr_pricing_wrapper
```
````

````{py:function} reduced_cost_arc_fixing(pricing_solver: typing.Any, master_lp_bound: float, incumbent_value: float, n_nodes: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], node_duals: typing.Dict[int, float], R: float, C: float, tol: float = 1e-06) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.reduced_cost_arc_fixing

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.reduced_cost_arc_fixing
```
````

````{py:function} apply_reduced_cost_edge_fixing(master: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.VRPPMasterProblem, pricing_solver: typing.Any, z_ub: float, z_lb: float) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.apply_reduced_cost_edge_fixing

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.pricing.smoothing.apply_reduced_cost_edge_fixing
```
````
