# {py:mod}`src.policies.old_bachelor_acceptance.solver`

```{py:module} src.policies.old_bachelor_acceptance.solver
```

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OBASolver <src.policies.old_bachelor_acceptance.solver.OBASolver>`
  - ```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver
    :summary:
    ```
````

### API

`````{py:class} OBASolver(*args, **kwargs)
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._accept

```{autodoc2-docstring} src.policies.old_bachelor_acceptance.solver.OBASolver._accept
```

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.old_bachelor_acceptance.solver.OBASolver._record_telemetry

````

`````
