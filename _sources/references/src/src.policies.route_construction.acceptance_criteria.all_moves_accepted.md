# {py:mod}`src.policies.route_construction.acceptance_criteria.all_moves_accepted`

```{py:module} src.policies.route_construction.acceptance_criteria.all_moves_accepted
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.all_moves_accepted
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AllMovesAccepted <src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted
    :summary:
    ```
````

### API

`````{py:class} AllMovesAccepted
:canonical: src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.all_moves_accepted.AllMovesAccepted.get_state

````

`````
