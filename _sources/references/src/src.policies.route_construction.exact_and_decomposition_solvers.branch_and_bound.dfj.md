# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_dfj_callback <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._dfj_callback>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._dfj_callback
    :summary:
    ```
* - {py:obj}`_setup_bb_model <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._setup_bb_model>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._setup_bb_model
    :summary:
    ```
* - {py:obj}`_extract_routes_from_adj <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._extract_routes_from_adj>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._extract_routes_from_adj
    :summary:
    ```
* - {py:obj}`run_bb_dfj <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj.run_bb_dfj>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj.run_bb_dfj
    :summary:
    ```
````

### API

````{py:function} _dfj_callback(model, where)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._dfj_callback

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._dfj_callback
```
````

````{py:function} _setup_bb_model(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], mandatory_indices: typing.Set[int], time_limit: float = 60.0, mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None) -> typing.Tuple[gurobipy.Model, typing.Dict[typing.Tuple[int, int], gurobipy.Var], typing.Dict[int, gurobipy.Var]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._setup_bb_model

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._setup_bb_model
```
````

````{py:function} _extract_routes_from_adj(adj: typing.Dict[int, typing.List[int]], num_nodes: int) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._extract_routes_from_adj

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj._extract_routes_from_adj
```
````

````{py:function} run_bb_dfj(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams] = None, mandatory_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj.run_bb_dfj

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dfj.run_bb_dfj
```
````
