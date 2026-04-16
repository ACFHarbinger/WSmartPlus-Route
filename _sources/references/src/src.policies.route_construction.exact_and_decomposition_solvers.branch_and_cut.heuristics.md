# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics
:allowtitles:
```

## Module Contents

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`construct_initial_solution <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_initial_solution>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_initial_solution
    :summary:
    ```
* - {py:obj}`construct_nn_solution <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_nn_solution>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_nn_solution
    :summary:
    ```
* - {py:obj}`farthest_insertion <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.farthest_insertion>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.farthest_insertion
    :summary:
    ```
* - {py:obj}`_routes_to_tour <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._routes_to_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._routes_to_tour
    :summary:
    ```
* - {py:obj}`_apply_2opt_to_tour <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._apply_2opt_to_tour>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._apply_2opt_to_tour
    :summary:
    ```
````

### API

````{py:function} construct_initial_solution(model: logic.src.policies.helpers.branching_solvers.vrpp_model.VRPPModel, seed: int = 42) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_initial_solution

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_initial_solution
```
````

````{py:function} construct_nn_solution(model: logic.src.policies.helpers.branching_solvers.vrpp_model.VRPPModel, seed: int = 42) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_nn_solution

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.construct_nn_solution
```
````

````{py:function} farthest_insertion(model: logic.src.policies.helpers.branching_solvers.vrpp_model.VRPPModel, profit_aware_operators: bool = False, expand_pool: bool = False) -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.farthest_insertion

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics.farthest_insertion
```
````

````{py:function} _routes_to_tour(routes: typing.List[typing.List[int]], depot: int) -> typing.List[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._routes_to_tour

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._routes_to_tour
```
````

````{py:function} _apply_2opt_to_tour(model: logic.src.policies.helpers.branching_solvers.vrpp_model.VRPPModel, tour: typing.List[int], max_iterations: int = 100) -> typing.List[int]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._apply_2opt_to_tour

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.heuristics._apply_2opt_to_tour
```
````
