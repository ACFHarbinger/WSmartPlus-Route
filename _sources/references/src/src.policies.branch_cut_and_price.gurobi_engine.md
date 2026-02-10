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

````{py:function} _setup_model(values, env)
:canonical: src.policies.branch_cut_and_price.gurobi_engine._setup_model

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._setup_model
```
````

````{py:function} _create_variables(model, nodes, customers, demands, capacity)
:canonical: src.policies.branch_cut_and_price.gurobi_engine._create_variables

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._create_variables
```
````

````{py:function} _create_objective(model, x, y, dist_matrix, demands, nodes, customers, R, C, must_go_indices)
:canonical: src.policies.branch_cut_and_price.gurobi_engine._create_objective

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._create_objective
```
````

````{py:function} _add_constraints(model, x, y, u, nodes, customers, demands, capacity)
:canonical: src.policies.branch_cut_and_price.gurobi_engine._add_constraints

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._add_constraints
```
````

````{py:function} _extract_solution(model, x, nodes)
:canonical: src.policies.branch_cut_and_price.gurobi_engine._extract_solution

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine._extract_solution
```
````

````{py:function} run_bcp_gurobi(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None)
:canonical: src.policies.branch_cut_and_price.gurobi_engine.run_bcp_gurobi

```{autodoc2-docstring} src.policies.branch_cut_and_price.gurobi_engine.run_bcp_gurobi
```
````
