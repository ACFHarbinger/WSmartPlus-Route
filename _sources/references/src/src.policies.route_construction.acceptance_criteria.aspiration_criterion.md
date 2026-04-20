# {py:mod}`src.policies.route_construction.acceptance_criteria.aspiration_criterion`

```{py:module} src.policies.route_construction.acceptance_criteria.aspiration_criterion
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.aspiration_criterion
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AspirationCriterion <src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion
    :summary:
    ```
````

### API

`````{py:class} AspirationCriterion()
:canonical: src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.aspiration_criterion.AspirationCriterion.get_state

````

`````
