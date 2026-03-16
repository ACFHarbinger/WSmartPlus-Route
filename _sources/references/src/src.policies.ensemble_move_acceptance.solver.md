# {py:mod}`src.policies.ensemble_move_acceptance.solver`

```{py:module} src.policies.ensemble_move_acceptance.solver
```

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EMASolver <src.policies.ensemble_move_acceptance.solver.EMASolver>`
  - ```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver.EMASolver
    :summary:
    ```
````

### API

`````{py:class} EMASolver(*args, **kwargs)
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver

Bases: {py:obj}`src.policies.base.base_acceptance_criteria.BaseAcceptanceSolver`

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver.EMASolver
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver.EMASolver.__init__
```

````{py:method} _accept(new_profit: float, current_profit: float, iteration: int) -> bool
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver._accept

````

````{py:method} _update_state(iteration: int)
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver._update_state

````

````{py:method} _check_sa(new_profit, current_profit) -> bool
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver._check_sa

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver.EMASolver._check_sa
```

````

````{py:method} _check_gd(new_profit, current_profit, iteration) -> bool
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver._check_gd

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver.EMASolver._check_gd
```

````

````{py:method} _check_ta(new_profit, current_profit, iteration) -> bool
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver._check_ta

```{autodoc2-docstring} src.policies.ensemble_move_acceptance.solver.EMASolver._check_ta
```

````

````{py:method} _record_telemetry(iteration: int, best_profit: float, current_profit: float)
:canonical: src.policies.ensemble_move_acceptance.solver.EMASolver._record_telemetry

````

`````
