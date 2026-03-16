# {py:mod}`src.policies.branch_price_cut.ortools_engine`

```{py:module} src.policies.branch_price_cut.ortools_engine
```

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bpc_ortools <src.policies.branch_price_cut.ortools_engine.run_bpc_ortools>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine.run_bpc_ortools
    :summary:
    ```
* - {py:obj}`_add_distance_constraints <src.policies.branch_price_cut.ortools_engine._add_distance_constraints>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._add_distance_constraints
    :summary:
    ```
* - {py:obj}`_add_capacity_constraints <src.policies.branch_price_cut.ortools_engine._add_capacity_constraints>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._add_capacity_constraints
    :summary:
    ```
* - {py:obj}`_add_waste_collecting_penalties <src.policies.branch_price_cut.ortools_engine._add_waste_collecting_penalties>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._add_waste_collecting_penalties
    :summary:
    ```
* - {py:obj}`_get_search_parameters <src.policies.branch_price_cut.ortools_engine._get_search_parameters>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._get_search_parameters
    :summary:
    ```
* - {py:obj}`_parse_routes <src.policies.branch_price_cut.ortools_engine._parse_routes>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._parse_routes
    :summary:
    ```
* - {py:obj}`_calculate_real_cost <src.policies.branch_price_cut.ortools_engine._calculate_real_cost>`
  - ```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._calculate_real_cost
    :summary:
    ```
````

### API

````{py:function} run_bpc_ortools(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_price_cut.ortools_engine.run_bpc_ortools

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine.run_bpc_ortools
```
````

````{py:function} _add_distance_constraints(routing: ortools.constraint_solver.pywrapcp.RoutingModel, manager: ortools.constraint_solver.pywrapcp.RoutingIndexManager, dist_matrix: numpy.ndarray) -> None
:canonical: src.policies.branch_price_cut.ortools_engine._add_distance_constraints

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._add_distance_constraints
```
````

````{py:function} _add_capacity_constraints(routing: ortools.constraint_solver.pywrapcp.RoutingModel, manager: ortools.constraint_solver.pywrapcp.RoutingIndexManager, wastes: typing.Dict[int, float], capacity: float, num_vehicles: int) -> None
:canonical: src.policies.branch_price_cut.ortools_engine._add_capacity_constraints

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._add_capacity_constraints
```
````

````{py:function} _add_waste_collecting_penalties(routing: ortools.constraint_solver.pywrapcp.RoutingModel, manager: ortools.constraint_solver.pywrapcp.RoutingIndexManager, wastes: typing.Dict[int, float], mandatory_nodes: typing.Optional[typing.List[int]], R: float, num_nodes: int) -> None
:canonical: src.policies.branch_price_cut.ortools_engine._add_waste_collecting_penalties

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._add_waste_collecting_penalties
```
````

````{py:function} _get_search_parameters(values: typing.Dict[str, typing.Any]) -> typing.Any
:canonical: src.policies.branch_price_cut.ortools_engine._get_search_parameters

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._get_search_parameters
```
````

````{py:function} _parse_routes(routing: ortools.constraint_solver.pywrapcp.RoutingModel, manager: ortools.constraint_solver.pywrapcp.RoutingIndexManager, solution: ortools.constraint_solver.pywrapcp.Assignment, num_vehicles: int) -> typing.List[typing.List[int]]
:canonical: src.policies.branch_price_cut.ortools_engine._parse_routes

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._parse_routes
```
````

````{py:function} _calculate_real_cost(routes: typing.List[typing.List[int]], dist_matrix: numpy.ndarray) -> float
:canonical: src.policies.branch_price_cut.ortools_engine._calculate_real_cost

```{autodoc2-docstring} src.policies.branch_price_cut.ortools_engine._calculate_real_cost
```
````
