# {py:mod}`src.policies.late_acceptance_hill_climbing.solver`

```{py:module} src.policies.late_acceptance_hill_climbing.solver
```

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LAHCSolver <src.policies.late_acceptance_hill_climbing.solver.LAHCSolver>`
  - ```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver
    :summary:
    ```
````

### API

`````{py:class} LAHCSolver(*args, **kwargs)
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._accept

```{autodoc2-docstring} src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._accept
```

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.late_acceptance_hill_climbing.solver.LAHCSolver._record_telemetry

````

`````
