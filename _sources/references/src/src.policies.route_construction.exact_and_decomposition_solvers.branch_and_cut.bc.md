# {py:mod}`src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc`

```{py:module} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndCutSolver <src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver
    :summary:
    ```
````

### API

`````{py:class} BranchAndCutSolver(model: logic.src.policies.helpers.solvers_and_matheuristics.vrpp_model.VRPPModel, params: typing.Optional[src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.params.BCParams] = None, scenarios: typing.Optional[typing.List[typing.Dict[int, float]]] = None, **kwargs: typing.Any)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver.solve

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver.solve
```

````

````{py:method} _build_initial_model()
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._build_initial_model

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._build_initial_model
```

````

````{py:method} _lazy_constraint_callback(model, where)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._lazy_constraint_callback

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._lazy_constraint_callback
```

````

````{py:method} _evaluate_pool_cuts(x_vals: numpy.ndarray, y_vals: numpy.ndarray) -> typing.List[typing.Any]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._evaluate_pool_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._evaluate_pool_cuts
```

````

````{py:method} _handle_custom_branching(model)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._handle_custom_branching

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._handle_custom_branching
```

````

````{py:method} _add_integer_cuts(model)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_integer_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_integer_cuts
```

````

````{py:method} _add_fractional_cuts(model)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_fractional_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_fractional_cuts
```

````

````{py:method} _add_pcsec_lazy(model, cut: logic.src.policies.helpers.solvers_and_matheuristics.PCSubtourEliminationCut)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_pcsec_lazy

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_pcsec_lazy
```

````

````{py:method} _add_capacity_cut_lazy(model, cut: logic.src.policies.helpers.solvers_and_matheuristics.CapacityCut)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_capacity_cut_lazy

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_capacity_cut_lazy
```

````

````{py:method} _add_pcsec_user(model, cut: logic.src.policies.helpers.solvers_and_matheuristics.PCSubtourEliminationCut)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_pcsec_user

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_pcsec_user
```

````

````{py:method} _add_capacity_cut_user(model, cut: logic.src.policies.helpers.solvers_and_matheuristics.CapacityCut)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_capacity_cut_user

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._add_capacity_cut_user
```

````

````{py:method} _separate_stochastic_capacity_cuts(model, x_vals: numpy.ndarray, y_vals: numpy.ndarray, is_integer: bool)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._separate_stochastic_capacity_cuts

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._separate_stochastic_capacity_cuts
```

````

````{py:method} _separate_multi_star_inequalities(model, x_vals: numpy.ndarray, y_vals: numpy.ndarray, is_integer: bool)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._separate_multi_star_inequalities

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._separate_multi_star_inequalities
```

````

````{py:method} _separate_lot_sizing_inequalities(model, x_vals: numpy.ndarray, y_vals: numpy.ndarray, is_integer: bool)
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._separate_lot_sizing_inequalities

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._separate_lot_sizing_inequalities
```

````

````{py:method} _set_start_solution(tour: typing.List[int])
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._set_start_solution

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._set_start_solution
```

````

````{py:method} _extract_solution() -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._extract_solution

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._extract_solution
```

````

````{py:method} _pre_optimize_lagrangian() -> typing.List[typing.Set[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._pre_optimize_lagrangian

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._pre_optimize_lagrangian
```

````

````{py:method} _find_k_tree_cycle(edges: typing.List[typing.Tuple[int, int]]) -> typing.Optional[typing.Set[int]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._find_k_tree_cycle

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._find_k_tree_cycle
```

````

````{py:method} _solve_k_tree(costs: typing.Dict[typing.Tuple[int, int], float]) -> typing.Tuple[typing.List[typing.Tuple[int, int]], float, typing.List[typing.Set[int]]]
:canonical: src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._solve_k_tree

```{autodoc2-docstring} src.policies.route_construction.exact_and_decomposition_solvers.branch_and_cut.bc.BranchAndCutSolver._solve_k_tree
```

````

`````
