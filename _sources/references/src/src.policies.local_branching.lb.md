# {py:mod}`src.policies.local_branching.lb`

```{py:module} src.policies.local_branching.lb
```

```{autodoc2-docstring} src.policies.local_branching.lb
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_add_local_branching_constraint <src.policies.local_branching.lb._add_local_branching_constraint>`
  - ```{autodoc2-docstring} src.policies.local_branching.lb._add_local_branching_constraint
    :summary:
    ```
* - {py:obj}`run_local_branching_gurobi <src.policies.local_branching.lb.run_local_branching_gurobi>`
  - ```{autodoc2-docstring} src.policies.local_branching.lb.run_local_branching_gurobi
    :summary:
    ```
````

### API

````{py:function} _add_local_branching_constraint(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], incumbent_x: typing.Dict[typing.Tuple[int, int], float], incumbent_y: typing.Dict[int, float], k: int) -> gurobipy.Constr
:canonical: src.policies.local_branching.lb._add_local_branching_constraint

```{autodoc2-docstring} src.policies.local_branching.lb._add_local_branching_constraint
```
````

````{py:function} run_local_branching_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], k: int = 10, max_iterations: int = 20, time_limit: float = 300.0, time_limit_per_iteration: float = 30.0, mip_limit_nodes: int = 5000, mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.local_branching.lb.run_local_branching_gurobi

```{autodoc2-docstring} src.policies.local_branching.lb.run_local_branching_gurobi
```
````
