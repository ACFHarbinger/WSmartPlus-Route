# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bb_optimizer <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher.run_bb_optimizer>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher.run_bb_optimizer
    :summary:
    ```
````

### API

````{py:function} run_bb_optimizer(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.params.BBParams] = None, mandatory_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher.run_bb_optimizer

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_bound.dispatcher.run_bb_optimizer
```
````
