# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IntegerLShapedEngine <src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine
    :summary:
    ```
````

### API

`````{py:class} IntegerLShapedEngine(model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, params: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.params.ILSBDParams)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine.__init__
```

````{py:method} solve(tree: typing.Any, demand_matrix: typing.Optional[numpy.ndarray] = None, bin_capacities: typing.Optional[numpy.ndarray] = None, initial_inventory: typing.Optional[numpy.ndarray] = None) -> typing.Tuple[typing.List[typing.List[int]], typing.Dict[int, float], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine.solve
```

````

````{py:method} _compute_deterministic_profit(routes: typing.List[typing.List[int]], y_hat: typing.Dict[int, float]) -> float
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine._compute_deterministic_profit

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.integer_l_shaped_benders_decomposition.ils_bd_engine.IntegerLShapedEngine._compute_deterministic_profit
```

````

`````
