# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`InventoryMasterProblem <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem
    :summary:
    ```
````

### API

`````{py:class} InventoryMasterProblem(model: logic.src.policies.helpers.branching_solvers.vrpp_model.VRPPModel, params: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams, horizon: int = 7, demand_matrix: typing.Optional[numpy.ndarray] = None, bin_capacities: typing.Optional[numpy.ndarray] = None, initial_inventory: typing.Optional[numpy.ndarray] = None, stockout_penalty: float = 500.0, big_m: float = 10000.0)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem

Bases: {py:obj}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.master_problem.MasterProblem`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.__init__
```

````{py:method} build_inventory() -> None
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.build_inventory

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.build_inventory
```

````

````{py:method} get_inventory_plan() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.get_inventory_plan

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.get_inventory_plan
```

````

````{py:method} get_collection_plan() -> typing.Dict[typing.Tuple[int, int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.get_collection_plan

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem.get_collection_plan
```

````

`````
