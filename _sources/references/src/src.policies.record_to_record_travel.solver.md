# {py:mod}`src.policies.record_to_record_travel.solver`

```{py:module} src.policies.record_to_record_travel.solver
```

```{autodoc2-docstring} src.policies.record_to_record_travel.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RRSolver <src.policies.record_to_record_travel.solver.RRSolver>`
  - ```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver
    :summary:
    ```
````

### API

`````{py:class} RRSolver(*args, **kwargs)
:canonical: src.policies.record_to_record_travel.solver.RRSolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.record_to_record_travel.solver.RRSolver._accept

```{autodoc2-docstring} src.policies.record_to_record_travel.solver.RRSolver._accept
```

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.record_to_record_travel.solver.RRSolver._record_telemetry

````

`````
