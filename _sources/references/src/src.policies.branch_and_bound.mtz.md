# {py:mod}`src.policies.branch_and_bound.mtz`

```{py:module} src.policies.branch_and_bound.mtz
```

```{autodoc2-docstring} src.policies.branch_and_bound.mtz
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BBSolver <src.policies.branch_and_bound.mtz.BBSolver>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`run_bb_mtz <src.policies.branch_and_bound.mtz.run_bb_mtz>`
  - ```{autodoc2-docstring} src.policies.branch_and_bound.mtz.run_bb_mtz
    :summary:
    ```
````

### API

`````{py:class} BBSolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, time_limit: float = 60.0, mip_gap: float = 0.01, branching_strategy: str = 'strong', strong_branching_limit: int = 5, must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None)
:canonical: src.policies.branch_and_bound.mtz.BBSolver

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver.__init__
```

````{py:method} _initialize_base_model() -> None
:canonical: src.policies.branch_and_bound.mtz.BBSolver._initialize_base_model

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._initialize_base_model
```

````

````{py:method} _evaluate_node(node: src.policies.branch_and_bound.node.Node) -> float
:canonical: src.policies.branch_and_bound.mtz.BBSolver._evaluate_node

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._evaluate_node
```

````

````{py:method} _is_integer(val: float, tol: float = 1e-06) -> bool
:canonical: src.policies.branch_and_bound.mtz.BBSolver._is_integer

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._is_integer
```

````

````{py:method} _get_branching_variable() -> typing.Optional[typing.Tuple[str, typing.Any]]
:canonical: src.policies.branch_and_bound.mtz.BBSolver._get_branching_variable

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._get_branching_variable
```

````

````{py:method} _strong_branching(fractional_vars: typing.List[typing.Tuple]) -> typing.Tuple[str, typing.Any]
:canonical: src.policies.branch_and_bound.mtz.BBSolver._strong_branching

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._strong_branching
```

````

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.mtz.BBSolver.solve

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver.solve
```

````

````{py:method} _is_valid_solution(routes: typing.List[typing.List[int]]) -> bool
:canonical: src.policies.branch_and_bound.mtz.BBSolver._is_valid_solution

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._is_valid_solution
```

````

````{py:method} _extract_routes() -> typing.List[typing.List[int]]
:canonical: src.policies.branch_and_bound.mtz.BBSolver._extract_routes

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.BBSolver._extract_routes
```

````

`````

````{py:function} run_bb_mtz(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Optional[src.policies.branch_and_bound.params.BBParams] = None, must_go_indices: typing.Optional[typing.Set[int]] = None, env: typing.Optional[gurobipy.Env] = None, recorder: typing.Optional[logic.src.tracking.viz_mixin.PolicyStateRecorder] = None, **kwargs: typing.Any) -> typing.Tuple[typing.List[typing.List[int]], float]
:canonical: src.policies.branch_and_bound.mtz.run_bb_mtz

```{autodoc2-docstring} src.policies.branch_and_bound.mtz.run_bb_mtz
```
````
