# {py:mod}`src.policies.branch_and_bound.bb_simple`

```{py:module} src.policies.branch_and_bound.bb_simple
```

```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_dfj_callback <src.policies.branch_and_bound.bb_simple._dfj_callback>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple._dfj_callback
    :summary:
    ```
* - {py:obj}`_setup_bb_model <src.policies.branch_and_bound.bb_simple._setup_bb_model>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple._setup_bb_model
    :summary:
    ```
* - {py:obj}`_extract_routes_from_adj <src.policies.branch_and_bound.bb_simple._extract_routes_from_adj>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple._extract_routes_from_adj
    :summary:
    ```
* - {py:obj}`run_bb_simple <src.policies.branch_and_bound.bb_simple.run_bb_simple>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple.run_bb_simple
    :summary:
    ```
````

### API

````{py:function} _dfj_callback(model, where)
:canonical: src.policies.branch_and_bound.bb_simple._dfj_callback

```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple._dfj_callback
```
````

````{py:function} _setup_bb_model(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], must_go_indices: typing.Set[int], time_limit: float, mip_gap: float, seed: int, env: typing.Optional[gurobipy.Env] = None) -> typing.Tuple[gurobipy.Model, typing.Dict[typing.Tuple[int, int], gurobipy.Var], typing.Dict[int, gurobipy.Var]]
:canonical: src.policies.branch_and_bound.bb_simple._setup_bb_model

```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple._setup_bb_model
```
````

````{py:function} _extract_routes_from_adj(adj: typing.Dict[int, typing.List[int]], num_nodes: int) -> typing.List[typing.List[int]]
:canonical: src.policies.branch_and_bound.bb_simple._extract_routes_from_adj

```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple._extract_routes_from_adj
```
````

````{py:function} run_bb_simple(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.bb_simple.run_bb_simple

```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple.run_bb_simple
```
````
