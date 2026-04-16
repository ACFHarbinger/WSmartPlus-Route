# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MPIntegerLShapedEngine <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine
    :summary:
    ```
````

### API

`````{py:class} MPIntegerLShapedEngine(model: logic.src.policies.helpers.branching_solvers.vrpp_model.VRPPModel, params: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams, horizon: int = 7, demand_matrix: typing.Optional[numpy.ndarray] = None, bin_capacities: typing.Optional[numpy.ndarray] = None, initial_inventory: typing.Optional[numpy.ndarray] = None, stockout_penalty: float = 500.0, big_m: float = 10000.0)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine

Bases: {py:obj}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine`

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine.__init__
```

````{py:method} solve_stochastic(tree: typing.Optional[typing.Any] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.Dict[int, float], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine.solve_stochastic

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine.solve_stochastic
```

````

````{py:method} _run_inventory_benders_loop(master: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.inventory_master.InventoryMasterProblem, tree: typing.Optional[typing.Any]) -> typing.Tuple[typing.List[typing.List[int]], typing.Dict[int, float], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine._run_inventory_benders_loop

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.mp_ils_bd_engine.MPIntegerLShapedEngine._run_inventory_benders_loop
```

````

`````
