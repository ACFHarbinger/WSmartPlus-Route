# {py:mod}`src.policies.vehicle_routing_problem_with_profits.interface`

```{py:module} src.policies.vehicle_routing_problem_with_profits.interface
```

```{autodoc2-docstring} src.policies.vehicle_routing_problem_with_profits.interface
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_vrpp_optimizer <src.policies.vehicle_routing_problem_with_profits.interface.run_vrpp_optimizer>`
  - ```{autodoc2-docstring} src.policies.vehicle_routing_problem_with_profits.interface.run_vrpp_optimizer
    :summary:
    ```
````

### API

````{py:function} run_vrpp_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], param: float, media: numpy.typing.NDArray[numpy.float64], desviopadrao: numpy.typing.NDArray[numpy.float64], values: typing.Dict[str, float], binsids: typing.List[int], must_go: typing.List[int], env: typing.Optional[gurobipy.Env] = None, number_vehicles: int = 1, time_limit: int = 60, optimizer: str = 'gurobi', max_iter_no_improv: int = 10)
:canonical: src.policies.vehicle_routing_problem_with_profits.interface.run_vrpp_optimizer

```{autodoc2-docstring} src.policies.vehicle_routing_problem_with_profits.interface.run_vrpp_optimizer
```
````
