# {py:mod}`src.policies.branch_cut_and_price`

```{py:module} src.policies.branch_cut_and_price
```

```{autodoc2-docstring} src.policies.branch_cut_and_price
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bcp <src.policies.branch_cut_and_price.run_bcp>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price.run_bcp
    :summary:
    ```
* - {py:obj}`_run_bcp_ortools <src.policies.branch_cut_and_price._run_bcp_ortools>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price._run_bcp_ortools
    :summary:
    ```
* - {py:obj}`_run_bcp_vrpy <src.policies.branch_cut_and_price._run_bcp_vrpy>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price._run_bcp_vrpy
    :summary:
    ```
* - {py:obj}`_run_bcp_gurobi <src.policies.branch_cut_and_price._run_bcp_gurobi>`
  - ```{autodoc2-docstring} src.policies.branch_cut_and_price._run_bcp_gurobi
    :summary:
    ```
````

### API

````{py:function} run_bcp(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None)
:canonical: src.policies.branch_cut_and_price.run_bcp

```{autodoc2-docstring} src.policies.branch_cut_and_price.run_bcp
```
````

````{py:function} _run_bcp_ortools(dist_matrix, demands, capacity, R, C, values, must_go_indices=None)
:canonical: src.policies.branch_cut_and_price._run_bcp_ortools

```{autodoc2-docstring} src.policies.branch_cut_and_price._run_bcp_ortools
```
````

````{py:function} _run_bcp_vrpy(dist_matrix, demands, capacity, R, C, values)
:canonical: src.policies.branch_cut_and_price._run_bcp_vrpy

```{autodoc2-docstring} src.policies.branch_cut_and_price._run_bcp_vrpy
```
````

````{py:function} _run_bcp_gurobi(dist_matrix, demands, capacity, R, C, values, must_go_indices=None, env=None)
:canonical: src.policies.branch_cut_and_price._run_bcp_gurobi

```{autodoc2-docstring} src.policies.branch_cut_and_price._run_bcp_gurobi
```
````
