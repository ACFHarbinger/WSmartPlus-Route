# {py:mod}`src.policies.simulated_annealing.solver`

```{py:module} src.policies.simulated_annealing.solver
```

```{autodoc2-docstring} src.policies.simulated_annealing.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SASolver <src.policies.simulated_annealing.solver.SASolver>`
  - ```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver
    :summary:
    ```
````

### API

`````{py:class} SASolver(*args, **kwargs)
:canonical: src.policies.simulated_annealing.solver.SASolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.simulated_annealing.solver.SASolver._accept

```{autodoc2-docstring} src.policies.simulated_annealing.solver.SASolver._accept
```

````

````{py:method} _update_state(iteration: int)
:canonical: src.policies.simulated_annealing.solver.SASolver._update_state

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.simulated_annealing.solver.SASolver._record_telemetry

````

`````
