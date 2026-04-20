# {py:mod}`src.policies.route_construction.matheuristics.two_phase_kernel_search.solver`

```{py:module} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_PhaseIStats <src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_run_phase1 <src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._run_phase1>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._run_phase1
    :summary:
    ```
* - {py:obj}`_build_phase2_kernel <src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._build_phase2_kernel>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._build_phase2_kernel
    :summary:
    ```
* - {py:obj}`run_tpks_gurobi <src.policies.route_construction.matheuristics.two_phase_kernel_search.solver.run_tpks_gurobi>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver.run_tpks_gurobi
    :summary:
    ```
````

### API

`````{py:class} _PhaseIStats
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats
```

````{py:attribute} var_frequency
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.var_frequency
:type: typing.Dict[gurobipy.Var, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.var_frequency
```

````

````{py:attribute} var_obj_contribution
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.var_obj_contribution
:type: typing.Dict[gurobipy.Var, float]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.var_obj_contribution
```

````

````{py:attribute} phase1_best_obj
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.phase1_best_obj
:type: float
:value: >
   None

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.phase1_best_obj
```

````

````{py:attribute} phase1_used_vars
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.phase1_used_vars
:type: typing.Set[gurobipy.Var]
:value: >
   'field(...)'

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats.phase1_used_vars
```

````

`````

````{py:function} _run_phase1(model: gurobipy.Model, x: typing.Dict, y: typing.Dict, dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], params: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams, phase1_time: float) -> src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._run_phase1

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._run_phase1
```
````

````{py:function} _build_phase2_kernel(x: typing.Dict, y: typing.Dict, stats: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._PhaseIStats, params: src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams) -> typing.Tuple[typing.List[gurobipy.Var], typing.List[gurobipy.Var]]
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._build_phase2_kernel

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver._build_phase2_kernel
```
````

````{py:function} run_tpks_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], params: typing.Optional[src.policies.route_construction.matheuristics.two_phase_kernel_search.params.TPKSParams] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.matheuristics.two_phase_kernel_search.solver.run_tpks_gurobi

```{autodoc2-docstring} src.policies.route_construction.matheuristics.two_phase_kernel_search.solver.run_tpks_gurobi
```
````
