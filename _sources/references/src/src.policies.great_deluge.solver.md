# {py:mod}`src.policies.great_deluge.solver`

```{py:module} src.policies.great_deluge.solver
```

```{autodoc2-docstring} src.policies.great_deluge.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GDSolver <src.policies.great_deluge.solver.GDSolver>`
  - ```{autodoc2-docstring} src.policies.great_deluge.solver.GDSolver
    :summary:
    ```
````

### API

`````{py:class} GDSolver(*args, **kwargs)
:canonical: src.policies.great_deluge.solver.GDSolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.great_deluge.solver.GDSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.great_deluge.solver.GDSolver.__init__
```

````{py:method} _update_state(iteration: int)
:canonical: src.policies.great_deluge.solver.GDSolver._update_state

````

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.great_deluge.solver.GDSolver._accept

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.great_deluge.solver.GDSolver._record_telemetry

````

`````
