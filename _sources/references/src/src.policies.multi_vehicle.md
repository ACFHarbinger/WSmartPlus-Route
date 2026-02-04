# {py:mod}`src.policies.multi_vehicle`

```{py:module} src.policies.multi_vehicle
```

```{autodoc2-docstring} src.policies.multi_vehicle
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`find_routes <src.policies.multi_vehicle.find_routes>`
  - ```{autodoc2-docstring} src.policies.multi_vehicle.find_routes
    :summary:
    ```
* - {py:obj}`find_routes_ortools <src.policies.multi_vehicle.find_routes_ortools>`
  - ```{autodoc2-docstring} src.policies.multi_vehicle.find_routes_ortools
    :summary:
    ```
* - {py:obj}`get_solution_costs <src.policies.multi_vehicle.get_solution_costs>`
  - ```{autodoc2-docstring} src.policies.multi_vehicle.get_solution_costs
    :summary:
    ```
````

### API

````{py:function} find_routes(dist_mat, demands, max_caps, to_collect, n_vehicles, coords=None, depot=0, time_limit=2.0)
:canonical: src.policies.multi_vehicle.find_routes

```{autodoc2-docstring} src.policies.multi_vehicle.find_routes
```
````

````{py:function} find_routes_ortools(dist_mat, demands, max_caps, to_collect, n_vehicles, coords=None, depot=0, time_limit=2)
:canonical: src.policies.multi_vehicle.find_routes_ortools

```{autodoc2-docstring} src.policies.multi_vehicle.find_routes_ortools
```
````

````{py:function} get_solution_costs(demands: typing.List[int], n_vehicles: int, manager: ortools.constraint_solver.pywrapcp.RoutingIndexManager, routing: ortools.constraint_solver.pywrapcp.RoutingModel, solution: ortools.constraint_solver.pywrapcp.Assignment, distancesC: typing.List[typing.List[int]]) -> typing.Tuple[typing.List[typing.List[int]], typing.List[int], typing.List[int], typing.List[int]]
:canonical: src.policies.multi_vehicle.get_solution_costs

```{autodoc2-docstring} src.policies.multi_vehicle.get_solution_costs
```
````
