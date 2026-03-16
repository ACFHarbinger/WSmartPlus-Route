# {py:mod}`src.policies.kernel_search.solver`

```{py:module} src.policies.kernel_search.solver
```

```{autodoc2-docstring} src.policies.kernel_search.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_setup_ks_model <src.policies.kernel_search.solver._setup_ks_model>`
  - ```{autodoc2-docstring} src.policies.kernel_search.solver._setup_ks_model
    :summary:
    ```
* - {py:obj}`_get_partitioned_vars <src.policies.kernel_search.solver._get_partitioned_vars>`
  - ```{autodoc2-docstring} src.policies.kernel_search.solver._get_partitioned_vars
    :summary:
    ```
* - {py:obj}`_solve_ks_iterations <src.policies.kernel_search.solver._solve_ks_iterations>`
  - ```{autodoc2-docstring} src.policies.kernel_search.solver._solve_ks_iterations
    :summary:
    ```
* - {py:obj}`_reconstruct_tour <src.policies.kernel_search.solver._reconstruct_tour>`
  - ```{autodoc2-docstring} src.policies.kernel_search.solver._reconstruct_tour
    :summary:
    ```
* - {py:obj}`run_kernel_search_gurobi <src.policies.kernel_search.solver.run_kernel_search_gurobi>`
  - ```{autodoc2-docstring} src.policies.kernel_search.solver.run_kernel_search_gurobi
    :summary:
    ```
````

### API

````{py:function} _setup_ks_model(model: gurobipy.Model, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int]) -> typing.Tuple[typing.Dict[typing.Tuple[int, int], gurobipy.Var], typing.Dict[int, gurobipy.Var], typing.Dict[int, gurobipy.Var]]
:canonical: src.policies.kernel_search.solver._setup_ks_model

```{autodoc2-docstring} src.policies.kernel_search.solver._setup_ks_model
```
````

````{py:function} _get_partitioned_vars(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], initial_kernel_size: int, bucket_size: int) -> typing.Tuple[typing.List[gurobipy.Var], typing.List[gurobipy.Var], typing.List[typing.List[gurobipy.Var]]]
:canonical: src.policies.kernel_search.solver._get_partitioned_vars

```{autodoc2-docstring} src.policies.kernel_search.solver._get_partitioned_vars
```
````

````{py:function} _solve_ks_iterations(model: gurobipy.Model, kernel_vars: typing.List[gurobipy.Var], buckets: typing.List[typing.List[gurobipy.Var]], remaining_vars: typing.List[gurobipy.Var], time_limit: float, mip_limit_nodes: int) -> typing.Set[gurobipy.Var]
:canonical: src.policies.kernel_search.solver._solve_ks_iterations

```{autodoc2-docstring} src.policies.kernel_search.solver._solve_ks_iterations
```
````

````{py:function} _reconstruct_tour(num_nodes: int, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], dist_matrix: numpy.ndarray) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.kernel_search.solver._reconstruct_tour

```{autodoc2-docstring} src.policies.kernel_search.solver._reconstruct_tour
```
````

````{py:function} run_kernel_search_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], initial_kernel_size: int = 50, bucket_size: int = 20, max_buckets: int = 10, time_limit: float = 300.0, mip_limit_nodes: int = 5000, mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.kernel_search.solver.run_kernel_search_gurobi

```{autodoc2-docstring} src.policies.kernel_search.solver.run_kernel_search_gurobi
```
````
