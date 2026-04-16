# {py:mod}`src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance`

```{py:module} src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OldBachelorAcceptance <src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance
    :summary:
    ```
````

### API

`````{py:class} OldBachelorAcceptance(contraction: float, dilation: float)
:canonical: src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.old_bachelor_acceptance.OldBachelorAcceptance.get_state

````

`````
