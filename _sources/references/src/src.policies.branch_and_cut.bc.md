# {py:mod}`src.policies.branch_and_cut.bc`

```{py:module} src.policies.branch_and_cut.bc
```

```{autodoc2-docstring} src.policies.branch_and_cut.bc
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BranchAndCutSolver <src.policies.branch_and_cut.bc.BranchAndCutSolver>`
  - ```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver
    :summary:
    ```
````

### API

`````{py:class} BranchAndCutSolver(model: logic.src.policies.branch_and_cut.vrpp_model.VRPPModel, time_limit: float = 300.0, mip_gap: float = 0.01, max_cuts_per_round: int = 50, use_heuristics: bool = True, verbose: bool = False)
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[int], float, typing.Dict[str, typing.Any]]
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver.solve

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver.solve
```

````

````{py:method} _build_initial_model()
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._build_initial_model

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._build_initial_model
```

````

````{py:method} _lazy_constraint_callback(model, where)
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._lazy_constraint_callback

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._lazy_constraint_callback
```

````

````{py:method} _add_lazy_cuts(model)
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._add_lazy_cuts

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._add_lazy_cuts
```

````

````{py:method} _add_subtour_cut(model, cut: logic.src.policies.branch_and_cut.separation.SubtourEliminationCut)
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._add_subtour_cut

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._add_subtour_cut
```

````

````{py:method} _add_capacity_cut(model, cut: logic.src.policies.branch_and_cut.separation.CapacityCut)
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._add_capacity_cut

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._add_capacity_cut
```

````

````{py:method} _set_start_solution(tour: typing.List[int])
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._set_start_solution

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._set_start_solution
```

````

````{py:method} _extract_solution() -> typing.Tuple[typing.List[int], float]
:canonical: src.policies.branch_and_cut.bc.BranchAndCutSolver._extract_solution

```{autodoc2-docstring} src.policies.branch_and_cut.bc.BranchAndCutSolver._extract_solution
```

````

`````
