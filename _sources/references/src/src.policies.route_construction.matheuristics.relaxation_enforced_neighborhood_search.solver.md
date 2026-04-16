# {py:mod}`src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver`

```{py:module} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_setup_rens_model <src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._setup_rens_model>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._setup_rens_model
    :summary:
    ```
* - {py:obj}`_apply_restrictions <src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._apply_restrictions>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._apply_restrictions
    :summary:
    ```
* - {py:obj}`run_rens_gurobi <src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver.run_rens_gurobi>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver.run_rens_gurobi
    :summary:
    ```
````

### API

````{py:function} _setup_rens_model(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], seed: int, env: typing.Optional[gurobipy.Env]) -> typing.Tuple[gurobipy.Model, typing.Dict[typing.Tuple[int, int], gurobipy.Var], typing.Dict[int, gurobipy.Var]]
:canonical: src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._setup_rens_model

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._setup_rens_model
```
````

````{py:function} _apply_restrictions(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var]) -> None
:canonical: src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._apply_restrictions

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver._apply_restrictions
```
````

````{py:function} run_rens_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], time_limit: float = 60.0, lp_time_limit: float = 10.0, mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver.run_rens_gurobi

```{autodoc2-docstring} src.policies.route_construction.matheuristics.relaxation_enforced_neighborhood_search.solver.run_rens_gurobi
```
````
