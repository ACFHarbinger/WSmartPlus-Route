# {py:mod}`src.policies.acceptance_criteria.binary_tournament_acceptance`

```{py:module} src.policies.acceptance_criteria.binary_tournament_acceptance
```

```{autodoc2-docstring} src.policies.acceptance_criteria.binary_tournament_acceptance
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BinaryTournamentAcceptance <src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance
    :summary:
    ```
````

### API

`````{py:class} BinaryTournamentAcceptance(p: float = 0.8, seed: typing.Optional[int] = None)
:canonical: src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.binary_tournament_acceptance.BinaryTournamentAcceptance.get_state

````

`````
