# {py:mod}`src.policies.acceptance_criteria.exponential_monte_carlo_counter`

```{py:module} src.policies.acceptance_criteria.exponential_monte_carlo_counter
```

```{autodoc2-docstring} src.policies.acceptance_criteria.exponential_monte_carlo_counter
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EMCQAcceptance <src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance
    :summary:
    ```
````

### API

`````{py:class} EMCQAcceptance(p: float = 0.05, p_boost: float = 0.5, q_threshold: typing.Union[int, typing.Callable[[], int]] = 100, seed: typing.Optional[int] = None, maximization: bool = False)
:canonical: src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance.setup

````

````{py:method} _get_q_threshold() -> int
:canonical: src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance._get_q_threshold

```{autodoc2-docstring} src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance._get_q_threshold
```

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.exponential_monte_carlo_counter.EMCQAcceptance.get_state

````

`````
