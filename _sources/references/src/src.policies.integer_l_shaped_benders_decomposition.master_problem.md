# {py:mod}`src.policies.integer_l_shaped_benders_decomposition.master_problem`

```{py:module} src.policies.integer_l_shaped_benders_decomposition.master_problem
```

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MasterProblem <src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem>`
  - ```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem
    :summary:
    ```
````

### API

`````{py:class} MasterProblem(model: logic.src.policies.other.branching_solvers.vrpp_model.VRPPModel, params: src.policies.integer_l_shaped_benders_decomposition.params.ILSBDParams)
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.__init__
```

````{py:method} build() -> None
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.build

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.build
```

````

````{py:method} add_optimality_cut(e: float, d: typing.Dict[int, float]) -> None
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.add_optimality_cut

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.add_optimality_cut
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], typing.Dict[int, float], float, float]
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.solve

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem.solve
```

````

````{py:method} _lazy_constraint_callback(model: typing.Any, where: int) -> None
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._lazy_constraint_callback

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._lazy_constraint_callback
```

````

````{py:method} _add_integer_cuts(model: typing.Any) -> None
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_integer_cuts

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_integer_cuts
```

````

````{py:method} _add_fractional_cuts(model: typing.Any) -> None
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_fractional_cuts

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._add_fractional_cuts
```

````

````{py:method} _extract_routes() -> typing.List[typing.List[int]]
:canonical: src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._extract_routes

```{autodoc2-docstring} src.policies.integer_l_shaped_benders_decomposition.master_problem.MasterProblem._extract_routes
```

````

`````
