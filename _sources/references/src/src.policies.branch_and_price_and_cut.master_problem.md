# {py:mod}`src.policies.branch_and_price_and_cut.master_problem`

```{py:module} src.policies.branch_and_price_and_cut.master_problem
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Route <src.policies.branch_and_price_and_cut.master_problem.Route>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.Route
    :summary:
    ```
* - {py:obj}`GlobalCutPool <src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool
    :summary:
    ```
* - {py:obj}`VRPPMasterProblem <src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.branch_and_price_and_cut.master_problem.logger>`
  - ```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.branch_and_price_and_cut.master_problem.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.logger
```

````

`````{py:class} Route(nodes: typing.List[int], cost: float, revenue: float, load: float, node_coverage: typing.Set[int])
:canonical: src.policies.branch_and_price_and_cut.master_problem.Route

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.Route
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.Route.__init__
```

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price_and_cut.master_problem.Route.__repr__

````

`````

`````{py:class} GlobalCutPool()
:canonical: src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool.__init__
```

````{py:method} add_cut(cut_type: str, data: typing.Any) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool.add_cut

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool.add_cut
```

````

````{py:method} apply_to_master(master: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem) -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool.apply_to_master

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool.apply_to_master
```

````

`````

`````{py:class} VRPPMasterProblem(n_nodes: int, mandatory_nodes: typing.Set[int], cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, vehicle_limit: typing.Optional[int] = None, global_cut_pool: typing.Optional[src.policies.branch_and_price_and_cut.master_problem.GlobalCutPool] = None)
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.__init__
```

````{py:method} add_symmetry_breaking_constraints() -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_symmetry_breaking_constraints

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_symmetry_breaking_constraints
```

````

````{py:method} purge_useless_columns(tolerance: float = -0.1) -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.purge_useless_columns

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.purge_useless_columns
```

````

````{py:method} sift_global_column_pool(node_duals: typing.Dict[int, float], rcc_duals: typing.Dict[typing.FrozenSet[int], float], sri_duals: typing.Dict[typing.FrozenSet[int], float], edge_clique_duals: typing.Dict[typing.Tuple[int, int], float], lci_duals: typing.Optional[typing.Dict[typing.FrozenSet[int], float]] = None, lci_node_alphas: typing.Optional[typing.Dict[typing.FrozenSet[int], typing.Dict[int, float]]] = None, branching_constraints: typing.Optional[typing.List[src.policies.branch_and_price_and_cut.branching.AnyBranchingConstraint]] = None, rc_tolerance: float = 1e-05) -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.sift_global_column_pool

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.sift_global_column_pool
```

````

````{py:method} calculate_reduced_cost(route: src.policies.branch_and_price_and_cut.master_problem.Route, dual_values: typing.Dict[str, typing.Any]) -> float
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.calculate_reduced_cost

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.calculate_reduced_cost
```

````

````{py:method} get_dual_values() -> typing.Dict[str, typing.Any]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_dual_values

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_dual_values
```

````

````{py:method} set_phase(phase: int) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.set_phase

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.set_phase
```

````

````{py:method} add_route_as_column(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_route_as_column

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_route_as_column
```

````

````{py:method} add_route(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_route

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_route
```

````

````{py:method} _add_column_to_model(route: src.policies.branch_and_price_and_cut.master_problem.Route) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._add_column_to_model

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._add_column_to_model
```

````

````{py:method} _wire_route_into_active_cuts(route: src.policies.branch_and_price_and_cut.master_problem.Route, var: typing.Any) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._wire_route_into_active_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._wire_route_into_active_cuts
```

````

````{py:method} build_model(initial_routes: typing.Optional[typing.List[src.policies.branch_and_price_and_cut.master_problem.Route]] = None) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.build_model

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.build_model
```

````

````{py:method} solve_lp_relaxation() -> typing.Tuple[typing.Optional[float], typing.Dict[int, float]]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.solve_lp_relaxation

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.solve_lp_relaxation
```

````

````{py:method} _apply_dual_smoothing() -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._apply_dual_smoothing

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._apply_dual_smoothing
```

````

````{py:method} solve_ip() -> typing.Tuple[float, typing.List[src.policies.branch_and_price_and_cut.master_problem.Route]]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.solve_ip

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.solve_ip
```

````

````{py:method} deduplicate_column_pool(tol: float = 1e-06) -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.deduplicate_column_pool

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.deduplicate_column_pool
```

````

````{py:method} get_reduced_cost_coefficients() -> typing.Dict[str, typing.Dict[typing.Union[int, typing.FrozenSet[int], str, typing.Tuple[int, int]], float]]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_reduced_cost_coefficients

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_reduced_cost_coefficients
```

````

````{py:method} get_node_visitation() -> typing.Dict[int, float]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_node_visitation

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_node_visitation
```

````

````{py:method} add_edge_clique_cut(u: int, v: int, coefficients: typing.Optional[typing.Dict[int, float]] = None, rhs: float = 1.0) -> bool
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_edge_clique_cut

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_edge_clique_cut
```

````

````{py:method} get_edge_usage() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_edge_usage

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_edge_usage
```

````

````{py:method} get_elementary_edge_usage() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_elementary_edge_usage

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.get_elementary_edge_usage
```

````

````{py:method} _aggregate_edge_usage(only_elementary: bool) -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._aggregate_edge_usage

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._aggregate_edge_usage
```

````

````{py:method} add_subset_row_cut(node_set: typing.Union[typing.List[int], typing.Set[int], typing.FrozenSet[int]]) -> bool
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_subset_row_cut

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_subset_row_cut
```

````

````{py:method} add_capacity_cut(node_list: typing.List[int], rhs: float, coefficients: typing.Optional[typing.Dict[int, float]] = None, is_global: bool = True, _skip_pool: bool = False) -> bool
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_capacity_cut

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_capacity_cut
```

````

````{py:method} add_lci_cut(node_list: typing.List[int], rhs: float, coefficients: typing.Dict[int, float], node_alphas: typing.Optional[typing.Dict[int, float]] = None, arc: typing.Optional[typing.Tuple[int, int]] = None) -> bool
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_lci_cut

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_lci_cut
```

````

````{py:method} add_sec_cut(node_list: typing.Union[typing.List[int], typing.Set[int], typing.FrozenSet[int]], rhs: float, cut_name: str = '', global_cut: bool = True, node_i: int = -1, node_j: int = -1, facet_form: str = '2.1') -> bool
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_sec_cut

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.add_sec_cut
```

````

````{py:method} remove_local_cuts() -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.remove_local_cuts

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.remove_local_cuts
```

````

````{py:method} _count_crossings(route: src.policies.branch_and_price_and_cut.master_problem.Route, node_set: typing.FrozenSet[int]) -> int
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._count_crossings

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem._count_crossings
```

````

````{py:method} has_artificial_variables_active(tol: float = 1e-06) -> bool
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.has_artificial_variables_active

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.has_artificial_variables_active
```

````

````{py:method} save_basis() -> typing.Optional[typing.Tuple[typing.List[int], typing.List[int]]]
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.save_basis

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.save_basis
```

````

````{py:method} restore_basis(vbasis: typing.List[int], cbasis: typing.List[int]) -> None
:canonical: src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.restore_basis

```{autodoc2-docstring} src.policies.branch_and_price_and_cut.master_problem.VRPPMasterProblem.restore_basis
```

````

`````
