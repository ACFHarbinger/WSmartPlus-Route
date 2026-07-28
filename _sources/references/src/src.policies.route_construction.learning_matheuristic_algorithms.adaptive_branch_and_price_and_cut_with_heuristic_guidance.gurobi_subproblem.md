# {py:mod}`src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem`

```{py:module} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GurobiVRPSubproblem <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`logger <src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.logger>`
  - ```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.logger
    :summary:
    ```
````

### API

````{py:data} logger
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.logger
:value: >
   'getLogger(...)'

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.logger
```

````

`````{py:class} GurobiVRPSubproblem(n_bins: int, capacity: float, cost_per_unit: float, time_limit: float = 30.0)
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem.__init__
```

````{py:method} solve(z_bar_day: typing.Dict[int, int], prizes: typing.Dict[int, float], dist_matrix: numpy.ndarray, loads: typing.Optional[numpy.ndarray] = None, relax: bool = True) -> typing.Tuple[float, typing.Dict[int, float], typing.List[int]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem.solve

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem.solve
```

````

````{py:method} _solve_lp(z_bar_day: typing.Dict[int, int], prizes: typing.Dict[int, float], dist_matrix: numpy.ndarray, loads: typing.Optional[numpy.ndarray], eligible: typing.List[int]) -> typing.Tuple[float, typing.Dict[int, float], typing.List[int]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem._solve_lp

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem._solve_lp
```

````

````{py:method} _solve_mip_with_dual_recovery(z_bar_day: typing.Dict[int, int], prizes: typing.Dict[int, float], dist_matrix: numpy.ndarray, loads: typing.Optional[numpy.ndarray], eligible: typing.List[int]) -> typing.Tuple[float, typing.Dict[int, float], typing.List[int]]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem._solve_mip_with_dual_recovery

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem._solve_mip_with_dual_recovery
```

````

````{py:method} _nearest_neighbour_route(visited: typing.List[int], dist_matrix: numpy.ndarray) -> typing.List[int]
:canonical: src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem._nearest_neighbour_route

```{autodoc2-docstring} src.policies.route_construction.learning_matheuristic_algorithms.adaptive_branch_and_price_and_cut_with_heuristic_guidance.gurobi_subproblem.GurobiVRPSubproblem._nearest_neighbour_route
```

````

`````
