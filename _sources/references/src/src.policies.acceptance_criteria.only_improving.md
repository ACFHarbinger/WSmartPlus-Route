# {py:mod}`src.policies.acceptance_criteria.only_improving`

```{py:module} src.policies.acceptance_criteria.only_improving
```

```{autodoc2-docstring} src.policies.acceptance_criteria.only_improving
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`OnlyImproving <src.policies.acceptance_criteria.only_improving.OnlyImproving>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.only_improving.OnlyImproving
    :summary:
    ```
````

### API

`````{py:class} OnlyImproving
:canonical: src.policies.acceptance_criteria.only_improving.OnlyImproving

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.only_improving.OnlyImproving
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.only_improving.OnlyImproving.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.only_improving.OnlyImproving.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.only_improving.OnlyImproving.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.only_improving.OnlyImproving.get_state

````

`````
