# {py:mod}`src.policies.branch_and_price.master_problem`

```{py:module} src.policies.branch_and_price.master_problem
```

```{autodoc2-docstring} src.policies.branch_and_price.master_problem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`Route <src.policies.branch_and_price.master_problem.Route>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.master_problem.Route
    :summary:
    ```
* - {py:obj}`VRPPMasterProblem <src.policies.branch_and_price.master_problem.VRPPMasterProblem>`
  - ```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem
    :summary:
    ```
````

### API

`````{py:class} Route(nodes: typing.List[int], cost: float, revenue: float, load: float, node_coverage: typing.Set[int])
:canonical: src.policies.branch_and_price.master_problem.Route

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.Route
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.Route.__init__
```

````{py:method} __repr__() -> str
:canonical: src.policies.branch_and_price.master_problem.Route.__repr__

````

`````

`````{py:class} VRPPMasterProblem(n_nodes: int, mandatory_nodes: typing.Set[int], cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float, vehicle_limit: typing.Optional[int] = None)
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.__init__
```

````{py:method} remove_unpromising_columns(threshold: float = -10.0) -> int
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.remove_unpromising_columns

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.remove_unpromising_columns
```

````

````{py:method} add_route_as_column(route: src.policies.branch_and_price.master_problem.Route) -> None
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_route_as_column

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_route_as_column
```

````

````{py:method} add_route(route: src.policies.branch_and_price.master_problem.Route) -> None
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_route

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_route
```

````

````{py:method} _add_column_to_model(route: src.policies.branch_and_price.master_problem.Route) -> None
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem._add_column_to_model

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem._add_column_to_model
```

````

````{py:method} build_model(initial_routes: typing.Optional[typing.List[src.policies.branch_and_price.master_problem.Route]] = None) -> None
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.build_model

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.build_model
```

````

````{py:method} solve_lp_relaxation() -> typing.Tuple[float, typing.Dict[int, float]]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.solve_lp_relaxation

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.solve_lp_relaxation
```

````

````{py:method} solve_ip() -> typing.Tuple[float, typing.List[src.policies.branch_and_price.master_problem.Route]]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.solve_ip

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.solve_ip
```

````

````{py:method} get_reduced_cost_coefficients() -> typing.Dict[str, typing.Dict[typing.Union[int, frozenset[int], str, typing.Tuple[int, int]], float]]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_reduced_cost_coefficients

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_reduced_cost_coefficients
```

````

````{py:method} get_node_visitation() -> typing.Dict[int, float]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_node_visitation

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_node_visitation
```

````

````{py:method} add_edge_lci_cut(u: int, v: int) -> bool
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_edge_lci_cut

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_edge_lci_cut
```

````

````{py:method} get_edge_usage() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_edge_usage

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_edge_usage
```

````

````{py:method} add_subset_row_cut(node_set: typing.List[int]) -> bool
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_subset_row_cut

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_subset_row_cut
```

````

````{py:method} add_set_packing_capacity_cut(node_list: typing.List[int], rhs: float) -> bool
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_set_packing_capacity_cut

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_set_packing_capacity_cut
```

````

````{py:method} add_sec_cut(node_list: typing.List[int], rhs: float, cut_name: str = '', global_cut: bool = True) -> bool
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_sec_cut

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_sec_cut
```

````

````{py:method} remove_local_cuts() -> int
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.remove_local_cuts

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.remove_local_cuts
```

````

````{py:method} find_and_add_violated_rcc(route_values: typing.Dict[int, float], routes: typing.List[src.policies.branch_and_price.master_problem.Route], max_cuts: int = 5) -> int
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.find_and_add_violated_rcc

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.find_and_add_violated_rcc
```

````

````{py:method} _find_customer_components(arc_flow: typing.Dict[typing.Tuple[int, int], float]) -> typing.List[typing.Set[int]]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem._find_customer_components

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem._find_customer_components
```

````

````{py:method} _count_crossings(route: src.policies.branch_and_price.master_problem.Route, node_set: typing.FrozenSet[int]) -> int
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem._count_crossings

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem._count_crossings
```

````

````{py:method} has_artificial_variables_active(tol: float = 1e-06) -> bool
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.has_artificial_variables_active

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.has_artificial_variables_active
```

````

````{py:method} save_basis() -> typing.Optional[typing.Tuple[typing.List[int], typing.List[int]]]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.save_basis

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.save_basis
```

````

````{py:method} restore_basis(basis: typing.Optional[typing.Tuple[typing.List[int], typing.List[int]]]) -> None
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.restore_basis

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.restore_basis
```

````

`````
