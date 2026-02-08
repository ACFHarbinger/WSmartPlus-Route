# {py:mod}`src.policies.vehicle_routing_problem_with_profits.gurobi`

```{py:module} src.policies.vehicle_routing_problem_with_profits.gurobi
```

```{autodoc2-docstring} src.policies.vehicle_routing_problem_with_profits.gurobi
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_run_gurobi_optimizer <src.policies.vehicle_routing_problem_with_profits.gurobi._run_gurobi_optimizer>`
  - ```{autodoc2-docstring} src.policies.vehicle_routing_problem_with_profits.gurobi._run_gurobi_optimizer
    :summary:
    ```
````

### API

````{py:function} _run_gurobi_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], env: typing.Optional[gurobipy.Env], param: float, media: numpy.typing.NDArray[numpy.float64], desviopadrao: numpy.typing.NDArray[numpy.float64], values: typing.Dict[str, float], binsids: typing.List[int], must_go: typing.List[int], number_vehicles: int = 1, time_limit: int = 60)
:canonical: src.policies.vehicle_routing_problem_with_profits.gurobi._run_gurobi_optimizer

```{autodoc2-docstring} src.policies.vehicle_routing_problem_with_profits.gurobi._run_gurobi_optimizer
```
````
