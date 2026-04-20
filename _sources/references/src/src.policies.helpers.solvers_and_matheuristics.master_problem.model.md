# {py:mod}`src.policies.helpers.solvers_and_matheuristics.master_problem.model`

```{py:module} src.policies.helpers.solvers_and_matheuristics.master_problem.model
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`VRPPMasterProblem <src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.solvers_and_matheuristics.master_problem.model.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.logger
```

````

`````{py:class} VRPPMasterProblem(n_nodes: int, mandatory_nodes: typing.Set[int], cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, vehicle_limit: typing.Optional[int] = None, global_cut_pool: typing.Optional[logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool.GlobalCutPool] = None)
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem

Bases: {py:obj}`logic.src.policies.helpers.solvers_and_matheuristics.master_problem.constraints.VRPPMasterProblemConstraintsMixin`, {py:obj}`logic.src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin`, {py:obj}`logic.src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.__init__
```

````{py:method} build_model(initial_routes: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route]] = None) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.build_model

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.build_model
```

````

````{py:method} solve_lp_relaxation() -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.solve_lp_relaxation

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.solve_lp_relaxation
```

````

````{py:method} _handle_infeasibility() -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._handle_infeasibility

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._handle_infeasibility
```

````

````{py:method} _extract_duals() -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._extract_duals

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._extract_duals
```

````

````{py:method} _apply_dual_smoothing() -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._apply_dual_smoothing

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._apply_dual_smoothing
```

````

````{py:method} solve_ip() -> typing.Tuple[float, typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.solve_ip

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.solve_ip
```

````

````{py:method} add_route(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.add_route

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.add_route
```

````

````{py:method} _add_column_to_model(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._add_column_to_model

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._add_column_to_model
```

````

````{py:method} _wire_route_into_active_cuts(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route, var: typing.Any) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._wire_route_into_active_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem._wire_route_into_active_cuts
```

````

````{py:method} purge_useless_columns(tolerance: float = -0.1) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.purge_useless_columns

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.purge_useless_columns
```

````

````{py:method} get_reduced_cost_coefficients() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.get_reduced_cost_coefficients

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.get_reduced_cost_coefficients
```

````

````{py:method} get_node_visitation() -> typing.Dict[int, float]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.get_node_visitation

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.get_node_visitation
```

````

````{py:method} get_edge_usage(only_elementary: bool = False) -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.get_edge_usage

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.get_edge_usage
```

````

````{py:method} save_basis() -> typing.Optional[typing.Tuple[typing.List[int], typing.List[int]]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.save_basis

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.save_basis
```

````

````{py:method} restore_basis(vbasis: typing.List[int], cbasis: typing.List[int]) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.restore_basis

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.model.VRPPMasterProblem.restore_basis
```

````

`````
