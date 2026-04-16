# {py:mod}`src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance`

```{py:module} src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EnsembleAcceptance <src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance
    :summary:
    ```
````

### API

`````{py:class} EnsembleAcceptance(criteria: typing.List[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion], rule: str = 'G-VOT')
:canonical: src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.ensemble_move_acceptance.EnsembleAcceptance.get_state

````

`````
