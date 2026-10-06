# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_clarke_wright_trips <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._clarke_wright_trips>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._clarke_wright_trips
    :summary:
    ```
* - {py:obj}`_warm_starts <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._warm_starts>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._warm_starts
    :summary:
    ```
* - {py:obj}`_plan_objective <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._plan_objective>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._plan_objective
    :summary:
    ```
* - {py:obj}`_run_gurobi_optimizer <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._run_gurobi_optimizer>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._run_gurobi_optimizer
    :summary:
    ```
````

### API

````{py:function} _clarke_wright_trips(d: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData, distance_matrix: typing.List[typing.List[float]], visit: typing.List[int]) -> typing.Optional[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._clarke_wright_trips

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._clarke_wright_trips
```
````

````{py:function} _warm_starts(d: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData, distance_matrix: typing.List[typing.List[float]], forced_nodes: typing.List[int]) -> typing.List[typing.List[typing.List[int]]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._warm_starts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._warm_starts
```
````

````{py:function} _plan_objective(d: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData, distance_matrix: typing.List[typing.List[float]], trips: typing.List[typing.List[int]]) -> typing.Tuple[float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._plan_objective

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._plan_objective
```
````

````{py:function} _run_gurobi_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], env: typing.Optional[gurobipy.Env], values: typing.Dict[str, float], binsids: typing.List[int], mandatory: typing.List[int], number_vehicles: int = 1, time_limit: int = 60, seed: int = 42, dual_values: typing.Optional[typing.Dict[int, float]] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._run_gurobi_optimizer

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._run_gurobi_optimizer
```
````
