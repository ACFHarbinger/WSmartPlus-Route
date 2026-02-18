# {py:mod}`src.policies.branch_cut_and_price.gurobi_engine`

```{py:module} src.policies.branch_cut_and_price.gurobi_engine
```

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`_setup_model <src.policies.branch_cut_and_price.gurobi_engine._setup_model>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._setup_model
    :summary:
    ```
* - {py:obj}`_create_variables <src.policies.branch_cut_and_price.gurobi_engine._create_variables>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._create_variables
    :summary:
    ```
* - {py:obj}`_create_objective <src.policies.branch_cut_and_price.gurobi_engine._create_objective>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._create_objective
    :summary:
    ```
* - {py:obj}`_add_constraints <src.policies.branch_cut_and_price.gurobi_engine._add_constraints>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._add_constraints
    :summary:
    ```
* - {py:obj}`_extract_solution <src.policies.branch_cut_and_price.gurobi_engine._extract_solution>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._extract_solution
    :summary:
    ```
* - {py:obj}`run_bcp_gurobi <src.policies.branch_cut_and_price.gurobi_engine.run_bcp_gurobi>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine.run_bcp_gurobi
    :summary:
    ```
````

### API

````{py:function} _setup_model(values: typing.Dict[str, typing.Any], env: typing.Optional[gurobipy.Env]) -> gurobipy.Model
:canonical: src.policies.branch_cut_and_price.gurobi_engine._setup_model

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._setup_model
```
````

````{py:function} _create_variables(model: gurobipy.Model, nodes: typing.List[int], customers: typing.List[int], demands: typing.Dict[int, float], capacity: float) -> typing.Tuple[typing.Dict[typing.Tuple[int, int], gurobipy.Var], typing.Dict[int, gurobipy.Var], typing.Dict[int, gurobipy.Var]]
:canonical: src.policies.branch_cut_and_price.gurobi_engine._create_variables

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._create_variables
```
````

````{py:function} _create_objective(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], dist_matrix: typing.Any, demands: typing.Dict[int, float], nodes: typing.List[int], customers: typing.List[int], R: float, C: float, mandatory_nodes: typing.Optional[typing.Set[int]]) -> None
:canonical: src.policies.branch_cut_and_price.gurobi_engine._create_objective

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._create_objective
```
````

````{py:function} _add_constraints(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], y: typing.Dict[int, gurobipy.Var], u: typing.Dict[int, gurobipy.Var], nodes: typing.List[int], customers: typing.List[int], demands: typing.Dict[int, float], capacity: float) -> None
:canonical: src.policies.branch_cut_and_price.gurobi_engine._add_constraints

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._add_constraints
```
````

````{py:function} _extract_solution(model: gurobipy.Model, x: typing.Dict[typing.Tuple[int, int], gurobipy.Var], nodes: typing.List[int]) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_cut_and_price.gurobi_engine._extract_solution

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._extract_solution
```
````

````{py:function} run_bcp_gurobi(dist_matrix: typing.Any, demands: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], mandatory_nodes: typing.Optional[typing.List[int]] = None, env: typing.Optional[gurobipy.Env] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_cut_and_price.gurobi_engine.run_bcp_gurobi

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine.run_bcp_gurobi
```
````
