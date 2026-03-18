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

````{py:function} run_bb_simple(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.bb_simple.run_bb_simple

```{autodoc2-docstring} src.policies.branch_and_bound.bb_simple.run_bb_simple
```
````
