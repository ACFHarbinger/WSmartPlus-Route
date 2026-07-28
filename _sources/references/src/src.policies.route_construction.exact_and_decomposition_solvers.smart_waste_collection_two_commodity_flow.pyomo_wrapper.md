# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_run_pyomo_tcf_optimizer <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper._run_pyomo_tcf_optimizer>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper._run_pyomo_tcf_optimizer
    :summary:
    ```
````

### API

````{py:function} _run_pyomo_tcf_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], values: typing.Dict[str, float], binsids: typing.List[int], mandatory_nodes: typing.List[int], number_vehicles: int = 1, time_limit: int = 60, solver_id: str = 'scip', seed: int = 42, dual_values: typing.Optional[typing.Dict[int, float]] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper._run_pyomo_tcf_optimizer

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.pyomo_wrapper._run_pyomo_tcf_optimizer
```
````
