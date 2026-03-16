# {py:mod}`src.policies.step_counting_hill_climbing.solver`

```{py:module} src.policies.step_counting_hill_climbing.solver
```

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SCHCSolver <src.policies.step_counting_hill_climbing.solver.SCHCSolver>`
  - ```{autodoc2-docstring} src.policies.step_counting_hill_climbing.solver.SCHCSolver
    :summary:
    ```
````

### API

`````{py:class} SCHCSolver(*args, **kwargs)
:canonical: src.policies.step_counting_hill_climbing.solver.SCHCSolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.solver.SCHCSolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.solver.SCHCSolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.step_counting_hill_climbing.solver.SCHCSolver._accept

```{autodoc2-docstring} src.policies.step_counting_hill_climbing.solver.SCHCSolver._accept
```

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.step_counting_hill_climbing.solver.SCHCSolver._record_telemetry

````

`````
