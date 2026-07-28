# {py:mod}`src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing`

```{py:module} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing
```

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GeneralizedTsallisSA <src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA
    :summary:
    ```
````

### API

`````{py:class} GeneralizedTsallisSA(q: float = 1.5, p0: float = 0.5, window_size: int = 100, alpha: float = 0.95, min_temp: float = 1e-06, seed: typing.Optional[int] = None, maximization: bool = False)
:canonical: src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.setup

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.setup
```

````

````{py:method} _update_stats(delta: float) -> None
:canonical: src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA._update_stats

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA._update_stats
```

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.accept

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.accept
```

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.step

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.step
```

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.get_state

```{autodoc2-docstring} src.policies.acceptance_criteria.generalized_tsallis_simulated_annealing.GeneralizedTsallisSA.get_state
```

````

`````
