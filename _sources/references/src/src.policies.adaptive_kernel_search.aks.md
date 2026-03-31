# {py:mod}`src.policies.adaptive_kernel_search.aks`

```{py:module} src.policies.adaptive_kernel_search.aks
```

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_get_partitioned_vars_aks <src.policies.adaptive_kernel_search.aks._get_partitioned_vars_aks>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._get_partitioned_vars_aks
    :summary:
    ```
* - {py:obj}`_get_feasible <src.policies.adaptive_kernel_search.aks._get_feasible>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._get_feasible
    :summary:
    ```
* - {py:obj}`_assess_difficulty <src.policies.adaptive_kernel_search.aks._assess_difficulty>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._assess_difficulty
    :summary:
    ```
* - {py:obj}`_solve_easy_iterations <src.policies.adaptive_kernel_search.aks._solve_easy_iterations>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._solve_easy_iterations
    :summary:
    ```
* - {py:obj}`_solve_rigorous_iterations <src.policies.adaptive_kernel_search.aks._solve_rigorous_iterations>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._solve_rigorous_iterations
    :summary:
    ```
* - {py:obj}`_solve_aks_iterations <src.policies.adaptive_kernel_search.aks._solve_aks_iterations>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._solve_aks_iterations
    :summary:
    ```
* - {py:obj}`run_adaptive_kernel_search_gurobi <src.policies.adaptive_kernel_search.aks.run_adaptive_kernel_search_gurobi>`
  - ```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks.run_adaptive_kernel_search_gurobi
    :summary:
    ```
````

### API

````{py:function} _get_partitioned_vars_aks(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], initial_kernel_size: int, bucket_size: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int]) -> typing.Tuple[typing.List[gurobipy.Var], typing.List[gurobipy.Var], typing.List[typing.List[gurobipy.Var]]]
:canonical: src.policies.adaptive_kernel_search.aks._get_partitioned_vars_aks

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._get_partitioned_vars_aks
```
````

````{py:function} _get_feasible(model, remaining_vars, active_kernel)
:canonical: src.policies.adaptive_kernel_search.aks._get_feasible

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._get_feasible
```
````

````{py:function} _assess_difficulty(model, t_mip_k, t_easy, epsilon)
:canonical: src.policies.adaptive_kernel_search.aks._assess_difficulty

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._assess_difficulty
```
````

````{py:function} _solve_easy_iterations(model, remaining_vars, current_rem_idx, active_kernel, chunk_size, t_easy, x_h, best_obj)
:canonical: src.policies.adaptive_kernel_search.aks._solve_easy_iterations

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._solve_easy_iterations
```
````

````{py:function} _solve_rigorous_iterations(model, buckets, active_kernel, x_h, best_obj, max_buckets, time_limit, start_time, mip_limit_nodes)
:canonical: src.policies.adaptive_kernel_search.aks._solve_rigorous_iterations

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._solve_rigorous_iterations
```
````

````{py:function} _solve_aks_iterations(model: gurobipy.Model, kernel_vars: typing.List[gurobipy.Var], remaining_vars: typing.List[gurobipy.Var], bucket_size: int, max_buckets: int, time_limit: float, mip_limit_nodes: int, t_easy: float = 10.0, epsilon: float = 0.1) -> typing.Set[gurobipy.Var]
:canonical: src.policies.adaptive_kernel_search.aks._solve_aks_iterations

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks._solve_aks_iterations
```
````

````{py:function} run_adaptive_kernel_search_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], initial_kernel_size: int = 50, bucket_size: int = 20, max_buckets: int = 15, time_limit: float = 300.0, mip_limit_nodes: int = 10000, mip_gap: float = 0.01, t_easy: float = 10.0, epsilon: float = 0.1, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.adaptive_kernel_search.aks.run_adaptive_kernel_search_gurobi

```{autodoc2-docstring} src.policies.adaptive_kernel_search.aks.run_adaptive_kernel_search_gurobi
```
````
