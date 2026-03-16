# {py:mod}`src.policies.adaptive_kernel_search.solver`

```{py:module} src.policies.adaptive_kernel_search.solver
```

```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_partitioned_vars_aks <src.policies.adaptive_kernel_search.solver._get_partitioned_vars_aks>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver._get_partitioned_vars_aks
    :summary:
    ```
* - {py:obj}`_solve_aks_iterations <src.policies.adaptive_kernel_search.solver._solve_aks_iterations>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver._solve_aks_iterations
    :summary:
    ```
* - {py:obj}`run_adaptive_kernel_search_gurobi <src.policies.adaptive_kernel_search.solver.run_adaptive_kernel_search_gurobi>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver.run_adaptive_kernel_search_gurobi
    :summary:
    ```
````

### API

````{py:function} _get_partitioned_vars_aks(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], initial_kernel_size: int, bucket_size: int) -> typing.Tuple[typing.List[gurobipy.Var], typing.List[gurobipy.Var], typing.List[typing.List[gurobipy.Var]]]
:canonical: src.policies.adaptive_kernel_search.solver._get_partitioned_vars_aks

```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver._get_partitioned_vars_aks
```
````

````{py:function} _solve_aks_iterations(model: gurobipy.Model, kernel_vars: typing.List[gurobipy.Var], remaining_vars: typing.List[gurobipy.Var], initial_bucket_size: int, max_buckets: int, bucket_growth_factor: float, time_limit: float, mip_limit_nodes: int) -> typing.Set[gurobipy.Var]
:canonical: src.policies.adaptive_kernel_search.solver._solve_aks_iterations

```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver._solve_aks_iterations
```
````

````{py:function} run_adaptive_kernel_search_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], initial_kernel_size: int = 50, bucket_size: int = 20, max_buckets: int = 15, bucket_growth_factor: float = 1.2, time_limit: float = 300.0, mip_limit_nodes: int = 10000, mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.adaptive_kernel_search.solver.run_adaptive_kernel_search_gurobi

```{autodoc2-docstring} src.policies.adaptive_kernel_search.solver.run_adaptive_kernel_search_gurobi
```
````
