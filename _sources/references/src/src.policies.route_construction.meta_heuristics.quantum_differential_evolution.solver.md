# {py:mod}`src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver`

```{py:module} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`QDESolver <src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver>`
  - ```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver
    :summary:
    ```
````

### API

`````{py:class} QDESolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: logic.src.policies.route_construction.meta_heuristics.quantum_differential_evolution.params.QDEParams, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver.__init__
```

````{py:method} solve() -> typing.Tuple[typing.List[typing.List[int]], float, float]
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver.solve

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver.solve
```

````

````{py:method} _rotation_gate(xi: int, bi: int, worsening: bool, theta: float) -> float
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._rotation_gate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._rotation_gate
```

````

````{py:method} _collapse(thetas: numpy.ndarray) -> typing.List[typing.List[int]]
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._collapse

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._collapse
```

````

````{py:method} _evaluate(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._evaluate

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._evaluate
```

````

````{py:method} _cost(routes: typing.List[typing.List[int]]) -> float
:canonical: src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._cost

```{autodoc2-docstring} src.policies.route_construction.meta_heuristics.quantum_differential_evolution.solver.QDESolver._cost
```

````

`````
