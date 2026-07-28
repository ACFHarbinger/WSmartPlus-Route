# {py:mod}`src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns`

```{py:module} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns
```

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_shake_solution_gurobi <src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns._shake_solution_gurobi>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns._shake_solution_gurobi
    :summary:
    ```
* - {py:obj}`run_lb_vns_gurobi <src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns.run_lb_vns_gurobi>`
  - ```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns.run_lb_vns_gurobi
    :summary:
    ```
````

### API

````{py:function} _shake_solution_gurobi(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], incumbent_x: typing.Dict[typing.Tuple[int, int], float], incumbent_y: typing.Dict[int, float], k: int, seed: int = 42) -> typing.Tuple[typing.Optional[typing.Dict[typing.Tuple[int, int], float]], typing.Optional[typing.Dict[int, float]]]
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns._shake_solution_gurobi

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns._shake_solution_gurobi
```
````

````{py:function} run_lb_vns_gurobi(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, mandatory_nodes: typing.List[int], mip_gap: float = 0.01, seed: int = 42, env: typing.Optional[gurobipy.Env] = None, params: typing.Optional[logic.src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.params.LBVNSParams] = None) -> typing.Tuple[typing.List[int], float, float]
:canonical: src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns.run_lb_vns_gurobi

```{autodoc2-docstring} src.policies.route_construction.matheuristics.local_branching_variable_neighborhood_search.lb_vns.run_lb_vns_gurobi
```
````
