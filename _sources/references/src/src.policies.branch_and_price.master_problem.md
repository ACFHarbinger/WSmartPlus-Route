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

`````{py:class} VRPPMasterProblem(n_nodes: int, mandatory_nodes: typing.Set[int], cost_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, revenue_per_kg: float, cost_per_km: float)
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.__init__
```

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

````{py:method} add_capacity_cut(cut_nodes: typing.List[int], rhs: float) -> bool
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_capacity_cut

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.add_capacity_cut
```

````

````{py:method} get_edge_usage() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_edge_usage

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_edge_usage
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

````{py:method} get_reduced_cost_coefficients() -> typing.Dict[int, float]
:canonical: src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_reduced_cost_coefficients

```{autodoc2-docstring} src.policies.branch_and_price.master_problem.VRPPMasterProblem.get_reduced_cost_coefficients
```

````

`````
