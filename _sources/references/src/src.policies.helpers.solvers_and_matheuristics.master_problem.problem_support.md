# {py:mod}`src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support`

```{py:module} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MasterProblemSupport <src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport
    :summary:
    ```
* - {py:obj}`VRPPMasterProblemSupportMixin <src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.logger
```

````

`````{py:class} MasterProblemSupport
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport

Bases: {py:obj}`typing.Protocol`

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport
```

````{py:attribute} n_nodes
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.n_nodes
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.n_nodes
```

````

````{py:attribute} mandatory_nodes
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.mandatory_nodes
:type: typing.Set[int]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.mandatory_nodes
```

````

````{py:attribute} optional_nodes
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.optional_nodes
:type: typing.Set[int]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.optional_nodes
```

````

````{py:attribute} cost_matrix
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.cost_matrix
:type: numpy.ndarray[typing.Any, typing.Any]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.cost_matrix
```

````

````{py:attribute} wastes
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.wastes
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.wastes
```

````

````{py:attribute} capacity
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.capacity
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.capacity
```

````

````{py:attribute} R
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.R
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.R
```

````

````{py:attribute} C
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.C
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.C
```

````

````{py:attribute} vehicle_limit
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.vehicle_limit
:type: typing.Optional[int]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.vehicle_limit
```

````

````{py:attribute} global_cut_pool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.global_cut_pool
:type: logic.src.policies.helpers.solvers_and_matheuristics.master_problem.pool.GlobalCutPool
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.global_cut_pool
```

````

````{py:attribute} BIG_M
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.BIG_M
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.BIG_M
```

````

````{py:attribute} model
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.model
:type: typing.Optional[gurobipy.Model]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.model
```

````

````{py:attribute} routes
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.routes
:type: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.routes
```

````

````{py:attribute} lambda_vars
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.lambda_vars
:type: typing.List[gurobipy.Var]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.lambda_vars
```

````

````{py:attribute} dual_node_coverage
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_node_coverage
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_node_coverage
```

````

````{py:attribute} dual_vehicle_limit
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_vehicle_limit
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_vehicle_limit
```

````

````{py:attribute} global_column_pool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.global_column_pool
:type: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.global_column_pool
```

````

````{py:attribute} phase
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.phase
:type: int
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.phase
```

````

````{py:attribute} active_src_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_src_cuts
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_src_cuts
```

````

````{py:attribute} active_sec_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_sec_cuts
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_sec_cuts
```

````

````{py:attribute} active_sec_cuts_local
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_sec_cuts_local
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_sec_cuts_local
```

````

````{py:attribute} active_rcc_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_rcc_cuts
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_rcc_cuts
```

````

````{py:attribute} active_capacity_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_capacity_cuts
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_capacity_cuts
```

````

````{py:attribute} active_lci_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_lci_cuts
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_lci_cuts
```

````

````{py:attribute} active_lci_node_alphas
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_lci_node_alphas
:type: typing.Dict[typing.FrozenSet[int], typing.Dict[int, float]]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_lci_node_alphas
```

````

````{py:attribute} active_lci_arcs
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_lci_arcs
:type: typing.Dict[typing.FrozenSet[int], typing.Optional[typing.Tuple[int, int]]]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_lci_arcs
```

````

````{py:attribute} active_sri_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_sri_cuts
:type: typing.Dict[typing.FrozenSet[int], gurobipy.Constr]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_sri_cuts
```

````

````{py:attribute} active_edge_clique_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_edge_clique_cuts
:type: typing.Dict[typing.Tuple[int, int], typing.Tuple[gurobipy.Constr, typing.Dict[int, float]]]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.active_edge_clique_cuts
```

````

````{py:attribute} dual_src_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_src_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_src_cuts
```

````

````{py:attribute} dual_sec_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_sec_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_sec_cuts
```

````

````{py:attribute} dual_sec_cuts_local
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_sec_cuts_local
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_sec_cuts_local
```

````

````{py:attribute} dual_rcc_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_rcc_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_rcc_cuts
```

````

````{py:attribute} dual_capacity_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_capacity_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_capacity_cuts
```

````

````{py:attribute} dual_lci_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_lci_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_lci_cuts
```

````

````{py:attribute} dual_sri_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_sri_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_sri_cuts
```

````

````{py:attribute} dual_edge_clique_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_edge_clique_cuts
:type: typing.Dict[typing.Tuple[int, int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_edge_clique_cuts
```

````

````{py:attribute} dual_smoothing_alpha
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_smoothing_alpha
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.dual_smoothing_alpha
```

````

````{py:attribute} prev_dual_node_coverage
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_node_coverage
:type: typing.Dict[int, float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_node_coverage
```

````

````{py:attribute} prev_dual_vehicle_limit
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_vehicle_limit
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_vehicle_limit
```

````

````{py:attribute} prev_dual_capacity_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_capacity_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_capacity_cuts
```

````

````{py:attribute} prev_dual_sri_cuts
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_sri_cuts
:type: typing.Dict[typing.FrozenSet[int], float]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.prev_dual_sri_cuts
```

````

````{py:attribute} farkas_duals
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.farkas_duals
:type: typing.Dict[str, typing.Dict[typing.Any, float]]
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.farkas_duals
```

````

````{py:attribute} column_deletion_enabled
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.column_deletion_enabled
:type: bool
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.column_deletion_enabled
```

````

````{py:attribute} strict_set_partitioning
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.strict_set_partitioning
:type: bool
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.strict_set_partitioning
```

````

````{py:attribute} enable_dual_smoothing
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.enable_dual_smoothing
:type: bool
:value: >
   None

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.enable_dual_smoothing
```

````

````{py:method} build_model(initial_routes: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route]] = None) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.build_model

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.build_model
```

````

````{py:method} solve_lp_relaxation() -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.solve_lp_relaxation

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.solve_lp_relaxation
```

````

````{py:method} _handle_infeasibility() -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._handle_infeasibility

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._handle_infeasibility
```

````

````{py:method} _extract_duals() -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._extract_duals

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._extract_duals
```

````

````{py:method} _apply_dual_smoothing() -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._apply_dual_smoothing

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._apply_dual_smoothing
```

````

````{py:method} solve_ip() -> typing.Tuple[float, typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.solve_ip

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.solve_ip
```

````

````{py:method} add_route(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_route

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_route
```

````

````{py:method} _add_column_to_model(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._add_column_to_model

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._add_column_to_model
```

````

````{py:method} _wire_route_into_active_cuts(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route, var: typing.Any) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._wire_route_into_active_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._wire_route_into_active_cuts
```

````

````{py:method} purge_useless_columns(tolerance: float = -0.1) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.purge_useless_columns

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.purge_useless_columns
```

````

````{py:method} get_reduced_cost_coefficients() -> typing.Dict[str, typing.Any]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.get_reduced_cost_coefficients

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.get_reduced_cost_coefficients
```

````

````{py:method} get_node_visitation() -> typing.Dict[int, float]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.get_node_visitation

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.get_node_visitation
```

````

````{py:method} get_edge_usage(only_elementary: bool = False) -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.get_edge_usage

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.get_edge_usage
```

````

````{py:method} save_basis() -> typing.Optional[typing.Tuple[typing.List[int], typing.List[int]]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.save_basis

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.save_basis
```

````

````{py:method} restore_basis(vbasis: typing.List[int], cbasis: typing.List[int]) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.restore_basis

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.restore_basis
```

````

````{py:method} sift_global_column_pool(node_duals: typing.Dict[int, float], rcc_duals: typing.Dict[typing.FrozenSet[int], float], sri_duals: typing.Dict[typing.FrozenSet[int], float], edge_clique_duals: typing.Dict[typing.Tuple[int, int], float], lci_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, lci_node_alphas: typing.Optional[typing.Dict[typing.FrozenSet[int], typing.Dict[int, float]]] = None, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint]] = None, rc_tolerance: float = 1e-05) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.sift_global_column_pool

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.sift_global_column_pool
```

````

````{py:method} calculate_reduced_cost(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route, dual_values: typing.Dict[str, typing.Any]) -> float
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.calculate_reduced_cost

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.calculate_reduced_cost
```

````

````{py:method} set_phase(phase: int) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.set_phase

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.set_phase
```

````

````{py:method} deduplicate_column_pool(tol: float = 1e-06) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.deduplicate_column_pool

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.deduplicate_column_pool
```

````

````{py:method} has_artificial_variables_active(tol: float = 1e-06) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.has_artificial_variables_active

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.has_artificial_variables_active
```

````

````{py:method} add_edge_clique_cut(u: int, v: int, coefficients: typing.Optional[typing.Dict[int, float]] = None, rhs: float = 1.0) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_edge_clique_cut

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_edge_clique_cut
```

````

````{py:method} add_subset_row_cut(node_set: typing.Union[typing.List[int], typing.Set[int], typing.FrozenSet[int]]) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_subset_row_cut

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_subset_row_cut
```

````

````{py:method} add_capacity_cut(node_list: typing.List[int], rhs: float, coefficients: typing.Optional[typing.Dict[int, float]] = None, is_global: bool = True, _skip_pool: bool = False) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_capacity_cut

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_capacity_cut
```

````

````{py:method} add_lci_cut(node_list: typing.List[int], rhs: float, coefficients: typing.Dict[int, float], node_alphas: typing.Optional[typing.Dict[int, float]] = None, arc: typing.Optional[typing.Tuple[int, int]] = None) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_lci_cut

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_lci_cut
```

````

````{py:method} add_set_packing_capacity_cut(node_list: typing.List[int], rhs: float) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_set_packing_capacity_cut

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_set_packing_capacity_cut
```

````

````{py:method} add_sec_cut(node_list: typing.Union[typing.List[int], typing.Set[int], typing.FrozenSet[int]], rhs: float, cut_name: str = '', global_cut: bool = True, node_i: int = -1, node_j: int = -1, facet_form: str = '2.1') -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_sec_cut

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.add_sec_cut
```

````

````{py:method} _count_crossings(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route, node_set: typing.FrozenSet[int]) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._count_crossings

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._count_crossings
```

````

````{py:method} remove_local_cuts() -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.remove_local_cuts

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.remove_local_cuts
```

````

````{py:method} find_and_add_violated_rcc(route_values: typing.Dict[int, float], routes: typing.List[logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route], max_cuts: int = 5) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.find_and_add_violated_rcc

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport.find_and_add_violated_rcc
```

````

````{py:method} _find_customer_components(arc_flow: typing.Dict[typing.Tuple[int, int], float]) -> typing.List[typing.Set[int]]
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._find_customer_components

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.MasterProblemSupport._find_customer_components
```

````

`````

`````{py:class} VRPPMasterProblemSupportMixin
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin
```

````{py:method} sift_global_column_pool(node_duals: typing.Dict[int, float], rcc_duals: typing.Dict[typing.FrozenSet[int], float], sri_duals: typing.Dict[typing.FrozenSet[int], float], edge_clique_duals: typing.Dict[typing.Tuple[int, int], float], lci_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, lci_node_alphas: typing.Optional[typing.Dict[typing.FrozenSet[int], typing.Dict[int, float]]] = None, branching_constraints: typing.Optional[typing.List[logic.src.policies.helpers.solvers_and_matheuristics.branching.constraints.AnyBranchingConstraint]] = None, rc_tolerance: float = 1e-05) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.sift_global_column_pool

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.sift_global_column_pool
```

````

````{py:method} calculate_reduced_cost(route: logic.src.policies.helpers.solvers_and_matheuristics.common.route.Route, dual_values: typing.Dict[str, typing.Any]) -> float
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.calculate_reduced_cost

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.calculate_reduced_cost
```

````

````{py:method} set_phase(phase: int) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.set_phase

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.set_phase
```

````

````{py:method} deduplicate_column_pool(tol: float = 1e-06) -> int
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.deduplicate_column_pool

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.deduplicate_column_pool
```

````

````{py:method} has_artificial_variables_active(tol: float = 1e-06) -> bool
:canonical: src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.has_artificial_variables_active

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.master_problem.problem_support.VRPPMasterProblemSupportMixin.has_artificial_variables_active
```

````

`````
