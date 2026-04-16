# {py:mod}`src.policies.route_construction.acceptance_criteria.improving_and_equal`

```{py:module} src.policies.route_construction.acceptance_criteria.improving_and_equal
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.improving_and_equal
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ImprovingAndEqual <src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual
    :summary:
    ```
````

### API

`````{py:class} ImprovingAndEqual
:canonical: src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.improving_and_equal.ImprovingAndEqual.get_state

````

`````
