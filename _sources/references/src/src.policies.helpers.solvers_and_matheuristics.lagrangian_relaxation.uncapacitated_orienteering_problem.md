# {py:mod}`src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem`

```{py:module} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_op_dfj_callback <src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem._op_dfj_callback>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem._op_dfj_callback
    :summary:
    ```
* - {py:obj}`solve_uncapacitated_op <src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem.solve_uncapacitated_op>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem.solve_uncapacitated_op
    :summary:
    ```
````

### API

````{py:function} _op_dfj_callback(model: gurobipy.Model, where: int) -> None
:canonical: src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem._op_dfj_callback

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem._op_dfj_callback
```
````

````{py:function} solve_uncapacitated_op(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], lam: float, R: float, C: float, forced_in: typing.Optional[typing.Set[int]] = None, forced_out: typing.Optional[typing.Set[int]] = None, time_limit: float = 10.0, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.Set[int], float, float]
:canonical: src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem.solve_uncapacitated_op

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.uncapacitated_orienteering_problem.solve_uncapacitated_op
```
````
