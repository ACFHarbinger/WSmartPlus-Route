# {py:mod}`src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing`

```{py:module} src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`LateAcceptance <src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance
    :summary:
    ```
````

### API

`````{py:class} LateAcceptance(queue_size: int)
:canonical: src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.late_acceptance_hill_climbing.LateAcceptance.get_state

````

`````
