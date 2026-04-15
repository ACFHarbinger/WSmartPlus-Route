# {py:mod}`src.policies.branch_and_bound.lr_orienteering`

```{py:module} src.policies.branch_and_bound.lr_orienteering
```

```{autodoc2-docstring} src.policies.branch_and_bound.lr_orienteering
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_op_dfj_callback <src.policies.branch_and_bound.lr_orienteering._op_dfj_callback>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.lr_orienteering._op_dfj_callback
    :summary:
    ```
* - {py:obj}`solve_uncapacitated_op <src.policies.branch_and_bound.lr_orienteering.solve_uncapacitated_op>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.lr_orienteering.solve_uncapacitated_op
    :summary:
    ```
````

### API

````{py:function} _op_dfj_callback(model: gurobipy.Model, where: int) -> None
:canonical: src.policies.branch_and_bound.lr_orienteering._op_dfj_callback

```{autodoc2-docstring} src.policies.branch_and_bound.lr_orienteering._op_dfj_callback
```
````

````{py:function} solve_uncapacitated_op(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], lam: float, R: float, C: float, forced_in: typing.Optional[typing.Set[int]] = None, forced_out: typing.Optional[typing.Set[int]] = None, time_limit: float = 10.0, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.Set[int], float, float]
:canonical: src.policies.branch_and_bound.lr_orienteering.solve_uncapacitated_op

```{autodoc2-docstring} src.policies.branch_and_bound.lr_orienteering.solve_uncapacitated_op
```
````
