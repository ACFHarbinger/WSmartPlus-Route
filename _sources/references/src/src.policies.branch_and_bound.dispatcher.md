# {py:mod}`src.policies.branch_and_bound.dispatcher`

```{py:module} src.policies.branch_and_bound.dispatcher
```

```{autodoc2-docstring} src.policies.branch_and_bound.dispatcher
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bb_optimizer <src.policies.branch_and_bound.dispatcher.run_bb_optimizer>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.dispatcher.run_bb_optimizer
    :summary:
    ```
````

### API

````{py:function} run_bb_optimizer(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, seed: typing.Optional[int] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, formulation: str = 'dfj') -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.dispatcher.run_bb_optimizer

```{autodoc2-docstring} src.policies.branch_and_bound.dispatcher.run_bb_optimizer
```
````
