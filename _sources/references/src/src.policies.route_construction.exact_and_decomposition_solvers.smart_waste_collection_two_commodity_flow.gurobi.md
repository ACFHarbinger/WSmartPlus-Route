# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_Built <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built
    :summary:
    ```
````

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
* - {py:obj}`_flag <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._flag>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._flag
    :summary:
    ```
* - {py:obj}`_optional_float <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._optional_float>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._optional_float
    :summary:
    ```
* - {py:obj}`_build_paper <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_paper>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_paper
    :summary:
    ```
* - {py:obj}`_build_directed <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_directed>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_directed
    :summary:
    ```
* - {py:obj}`_options <src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._options>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._options
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

````{py:function} _flag(value: object) -> bool
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._flag

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._flag
```
````

````{py:function} _optional_float(value: object) -> typing.Optional[float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._optional_float

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._optional_float
```
````

````{py:class} _Built(x: typing.Any, g: typing.Any, k_var: typing.Any, forced: typing.List[typing.Any], set_starts: typing.Callable[[typing.List[typing.List[typing.List[int]]]], None], extract_route: typing.Callable[[], typing.List[int]])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built.__init__
```

````

````{py:function} _build_paper(mdl: gurobipy.Model, d: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData, distance_matrix: typing.List[typing.List[float]], forced_nodes: typing.List[int], number_vehicles: int, dual_values: typing.Optional[typing.Dict[int, float]], options: typing.Dict[str, typing.Any]) -> src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_paper

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_paper
```
````

````{py:function} _build_directed(mdl: gurobipy.Model, d: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow._tcf_data.TCFData, distance_matrix: typing.List[typing.List[float]], forced_nodes: typing.List[int], number_vehicles: int, dual_values: typing.Optional[typing.Dict[int, float]], options: typing.Dict[str, typing.Any]) -> src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._Built
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_directed

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._build_directed
```
````

````{py:function} _options(values: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._options

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._options
```
````

````{py:function} _run_gurobi_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], env: typing.Optional[gurobipy.Env], values: typing.Dict[str, typing.Any], binsids: typing.List[int], mandatory: typing.List[int], number_vehicles: int = 1, time_limit: int = 60, seed: int = 42, dual_values: typing.Optional[typing.Dict[int, float]] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._run_gurobi_optimizer

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.smart_waste_collection_two_commodity_flow.gurobi._run_gurobi_optimizer
```
````
