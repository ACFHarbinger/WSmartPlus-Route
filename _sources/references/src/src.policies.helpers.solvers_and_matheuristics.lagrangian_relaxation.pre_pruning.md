# {py:mod}`src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning`

```{py:module} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning
```

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`compute_lr_bound_at_node <src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.compute_lr_bound_at_node>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.compute_lr_bound_at_node
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.logger>`
  - ```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.logger
```

````

````{py:function} compute_lr_bound_at_node(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory: typing.Set[int], forced_out: typing.Set[int], params: typing.Any, time_budget: float, env: typing.Optional[typing.Any], recorder: typing.Optional[typing.Any]) -> typing.Tuple[float, float, typing.Set[int]]
:canonical: src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.compute_lr_bound_at_node

```{autodoc2-docstring} src.policies.helpers.solvers_and_matheuristics.lagrangian_relaxation.pre_pruning.compute_lr_bound_at_node
```
````
