# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GurobiMasterProblem <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.logger
```

````

`````{py:class} GurobiMasterProblem(n_bins: int, horizon: int, max_visits_per_bin: int = 1, theta_upper_bound: float = 1000000.0, mip_gap: float = 0.0001, time_limit: float = 60.0, output_flag: bool = False)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.__init__
```

````{py:method} _build(max_visits: int, theta_ub: float, mip_gap: float, time_limit: float, output_flag: bool) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem._build

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem._build
```

````

````{py:method} add_benders_cut(cut: typing.Dict) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.add_benders_cut

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.add_benders_cut
```

````

````{py:method} add_benders_cuts_bulk(cuts: typing.List[typing.Dict]) -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.add_benders_cuts_bulk

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.add_benders_cuts_bulk
```

````

````{py:method} solve() -> typing.Optional[typing.Dict[int, typing.Dict[int, int]]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.solve

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.solve
```

````

````{py:method} get_objective_value() -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.get_objective_value

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.get_objective_value
```

````

````{py:method} get_mip_gap() -> float
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.get_mip_gap

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.get_mip_gap
```

````

````{py:method} dispose() -> None
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.dispose

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_master.GurobiMasterProblem.dispose
```

````

`````
