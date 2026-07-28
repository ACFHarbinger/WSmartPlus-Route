# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MasterProblem <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem
    :summary:
    ```
* - {py:obj}`InventoryMasterProblem <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem
    :summary:
    ```
````

### API

`````{py:class} MasterProblem(model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, params: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.__init__
```

````{py:method} build() -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.build

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.build
```

````

````{py:method} add_optimality_cut(e: float, d: typing.Dict[int, float]) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.add_optimality_cut

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.add_optimality_cut
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], typing.Dict[int, float], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.solve
```

````

````{py:method} _lazy_constraint_callback(model: typing.Any, where: int) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._lazy_constraint_callback

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._lazy_constraint_callback
```

````

````{py:method} _add_integer_cuts(model: typing.Any) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_integer_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_integer_cuts
```

````

````{py:method} _add_fractional_cuts(model: typing.Any) -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_fractional_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_fractional_cuts
```

````

````{py:method} _extract_routes() -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._extract_routes

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._extract_routes
```

````

`````

`````{py:class} InventoryMasterProblem(model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, params: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams, horizon: int = 7, demand_matrix: typing.Optional[numpy.ndarray] = None, bin_capacities: typing.Optional[numpy.ndarray] = None, initial_inventory: typing.Optional[numpy.ndarray] = None, stockout_penalty: float = 500.0, big_m: float = 10000.0)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem

Bases: {py:obj}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.__init__
```

````{py:method} build_inventory() -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.build_inventory

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.build_inventory
```

````

````{py:method} get_inventory_plan() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.get_inventory_plan

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.get_inventory_plan
```

````

````{py:method} get_collection_plan() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.get_collection_plan

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.InventoryMasterProblem.get_collection_plan
```

````

`````
