# {py:mod}`src.policies.vrpp_optimizer`

```{py:module} src.policies.vrpp_optimizer
```

```{autodoc2-docstring} src.policies.vrpp_optimizer
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_vrpp_optimizer <src.policies.vrpp_optimizer.run_vrpp_optimizer>`
  - ```{autodoc2-docstring} src.policies.vrpp_optimizer.run_vrpp_optimizer
    :summary:
    ```
* - {py:obj}`_run_gurobi_optimizer <src.policies.vrpp_optimizer._run_gurobi_optimizer>`
  - ```{autodoc2-docstring} src.policies.vrpp_optimizer._run_gurobi_optimizer
    :summary:
    ```
* - {py:obj}`_run_hexaly_optimizer <src.policies.vrpp_optimizer._run_hexaly_optimizer>`
  - ```{autodoc2-docstring} src.policies.vrpp_optimizer._run_hexaly_optimizer
    :summary:
    ```
````

### API

````{py:function} run_vrpp_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], param: float, media: numpy.typing.NDArray[numpy.float64], desviopadrao: numpy.typing.NDArray[numpy.float64], values: typing.Dict[str, float], binsids: typing.List[int], must_go: typing.List[int], env: typing.Optional[gurobipy.Env] = None, number_vehicles: int = 1, time_limit: int = 60, optimizer: str = 'gurobi', max_iter_no_improv: int = 10)
:canonical: src.policies.vrpp_optimizer.run_vrpp_optimizer

```{autodoc2-docstring} src.policies.vrpp_optimizer.run_vrpp_optimizer
```
````

````{py:function} _run_gurobi_optimizer(bins: numpy.typing.NDArray[numpy.float64], distance_matrix: typing.List[typing.List[float]], env: typing.Optional[gurobipy.Env], param: float, media: numpy.typing.NDArray[numpy.float64], desviopadrao: numpy.typing.NDArray[numpy.float64], values: typing.Dict[str, float], binsids: typing.List[int], must_go: typing.List[int], number_vehicles: int = 1, time_limit: int = 60)
:canonical: src.policies.vrpp_optimizer._run_gurobi_optimizer

```{autodoc2-docstring} src.policies.vrpp_optimizer._run_gurobi_optimizer
```
````

````{py:function} _run_hexaly_optimizer(bins: numpy.typing.NDArray[numpy.float64], distancematrix: typing.List[typing.List[float]], param: float, media: numpy.typing.NDArray[numpy.float64], desviopadrao: numpy.typing.NDArray[numpy.float64], values: typing.Dict[str, float], must_go: typing.List[int], number_vehicles: int = 1, time_limit: int = 60, max_iter_no_improv: int = 10)
:canonical: src.policies.vrpp_optimizer._run_hexaly_optimizer

```{autodoc2-docstring} src.policies.vrpp_optimizer._run_hexaly_optimizer
```
````
