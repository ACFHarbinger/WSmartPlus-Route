# {py:mod}`src.policies.route_construction.matheuristics.kernel_search.solver`

```{py:module} src.policies.route_construction.matheuristics.kernel_search.solver
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_check_route_capacity <src.policies.route_construction.matheuristics.kernel_search.solver._check_route_capacity>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._check_route_capacity
    :summary:
    ```
* - {py:obj}`_dfj_subtour_elimination_callback <src.policies.route_construction.matheuristics.kernel_search.solver._dfj_subtour_elimination_callback>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._dfj_subtour_elimination_callback
    :summary:
    ```
* - {py:obj}`_setup_ks_model <src.policies.route_construction.matheuristics.kernel_search.solver._setup_ks_model>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._setup_ks_model
    :summary:
    ```
* - {py:obj}`_set_mip_start <src.policies.route_construction.matheuristics.kernel_search.solver._set_mip_start>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._set_mip_start
    :summary:
    ```
* - {py:obj}`_separate_fractional_subtours <src.policies.route_construction.matheuristics.kernel_search.solver._separate_fractional_subtours>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._separate_fractional_subtours
    :summary:
    ```
* - {py:obj}`_root_node_callback <src.policies.route_construction.matheuristics.kernel_search.solver._root_node_callback>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._root_node_callback
    :summary:
    ```
* - {py:obj}`_get_partitioned_vars <src.policies.route_construction.matheuristics.kernel_search.solver._get_partitioned_vars>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._get_partitioned_vars
    :summary:
    ```
* - {py:obj}`_solve_ks_iterations <src.policies.route_construction.matheuristics.kernel_search.solver._solve_ks_iterations>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._solve_ks_iterations
    :summary:
    ```
* - {py:obj}`_reconstruct_tour <src.policies.route_construction.matheuristics.kernel_search.solver._reconstruct_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._reconstruct_tour
    :summary:
    ```
* - {py:obj}`run_kernel_search_gurobi <src.policies.route_construction.matheuristics.kernel_search.solver.run_kernel_search_gurobi>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver.run_kernel_search_gurobi
    :summary:
    ```
````

### API

````{py:function} _check_route_capacity(model, G, x_vars, component)
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._check_route_capacity

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._check_route_capacity
```
````

````{py:function} _dfj_subtour_elimination_callback(model, where)
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._dfj_subtour_elimination_callback

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._dfj_subtour_elimination_callback
```
````

````{py:function} _setup_ks_model(model: gurobipy.Model, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], use_binary_vars: bool = False) -> typing.Tuple[typing.Dict[typing.Tuple[int, int], gurobipy.Var], typing.Dict[int, gurobipy.Var]]
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._setup_ks_model

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._setup_ks_model
```
````

````{py:function} _set_mip_start(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int])
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._set_mip_start

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._set_mip_start
```
````

````{py:function} _separate_fractional_subtours(model, x_vars, num_nodes)
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._separate_fractional_subtours

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._separate_fractional_subtours
```
````

````{py:function} _root_node_callback(model, where)
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._root_node_callback

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._root_node_callback
```
````

````{py:function} _get_partitioned_vars(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], initial_kernel_size: int, bucket_size: int, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int]) -> typing.Tuple[typing.List[gurobipy.Var], typing.List[gurobipy.Var], typing.List[typing.List[gurobipy.Var]]]
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._get_partitioned_vars

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._get_partitioned_vars
```
````

````{py:function} _solve_ks_iterations(model: gurobipy.Model, kernel_vars: typing.List[gurobipy.Var], buckets: typing.List[typing.List[gurobipy.Var]], remaining_vars: typing.List[gurobipy.Var], time_limit: float, mip_limit_nodes: int) -> typing.Set[gurobipy.Var]
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._solve_ks_iterations

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._solve_ks_iterations
```
````

````{py:function} _reconstruct_tour(num_nodes: int, x: typing.Dict[typing.Tuple[int, int], typing.Any], dist_matrix: numpy.ndarray) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver._reconstruct_tour

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver._reconstruct_tour
```
````

````{py:function} run_kernel_search_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], initial_kernel_size: int = 50, bucket_size: int = 20, max_buckets: int = 10, time_limit: float = 300.0, mip_limit_nodes: int = 5000, mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.matheuristics.kernel_search.solver.run_kernel_search_gurobi

```{autodoc2-docstring} src.policies.route_construction.matheuristics.kernel_search.solver.run_kernel_search_gurobi
```
````
