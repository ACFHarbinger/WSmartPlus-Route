# {py:mod}`src.policies.branch_and_bound.solver`

```{py:module} src.policies.branch_and_bound.solver
```

```{autodoc2-docstring} src.policies.branch_and_bound.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BBSolver <src.policies.branch_and_bound.solver.BBSolver>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bb <src.policies.branch_and_bound.solver.run_bb>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.solver.run_bb
    :summary:
    ```
````

### API

`````{py:class} BBSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None)
:canonical: src.policies.branch_and_bound.solver.BBSolver

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver.__init__
```

````{py:method} _setup_relaxation_model(node: src.policies.branch_and_bound.node.Node) -> typing.Tuple[gurobipy.Model, typing.Dict, typing.Dict, typing.Dict]
:canonical: src.policies.branch_and_bound.solver.BBSolver._setup_relaxation_model

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver._setup_relaxation_model
```

````

````{py:method} _is_integer(val: float, tol: float = 1e-06) -> bool
:canonical: src.policies.branch_and_bound.solver.BBSolver._is_integer

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver._is_integer
```

````

````{py:method} _get_branching_variable(x: typing.Dict, y: typing.Dict) -> typing.Optional[typing.Tuple[str, typing.Any]]
:canonical: src.policies.branch_and_bound.solver.BBSolver._get_branching_variable

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver._get_branching_variable
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.solver.BBSolver.solve

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver.solve
```

````

````{py:method} _extract_routes(x_vars: typing.Dict) -> typing.List[typing.List[int]]
:canonical: src.policies.branch_and_bound.solver.BBSolver._extract_routes

```{autodoc2-docstring} src.policies.branch_and_bound.solver.BBSolver._extract_routes
```

````

`````

````{py:function} run_bb(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, values: typing.Dict[str, typing.Any], must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.solver.run_bb

```{autodoc2-docstring} src.policies.branch_and_bound.solver.run_bb
```
````
