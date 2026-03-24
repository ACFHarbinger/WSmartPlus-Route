# {py:mod}`src.policies.threshold_accepting.solver`

```{py:module} src.policies.threshold_accepting.solver
```

```{autodoc2-docstring} src.policies.threshold_accepting.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`TASolver <src.policies.threshold_accepting.solver.TASolver>`
  - ```{autodoc2-docstring} src.policies.threshold_accepting.solver.TASolver
    :summary:
    ```
````

### API

`````{py:class} TASolver(dist_matrix: numpy.ndarray, wastes: typing.Dict[int, float], capacity: float, R: float, C: float, params: typing.Any, mandatory_nodes: typing.Optional[typing.List[int]] = None)
:canonical: src.policies.threshold_accepting.solver.TASolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.threshold_accepting.solver.TASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.threshold_accepting.solver.TASolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.threshold_accepting.solver.TASolver._accept

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.threshold_accepting.solver.TASolver._record_telemetry

````

`````
