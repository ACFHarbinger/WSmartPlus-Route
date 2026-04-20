# {py:mod}`src.policies.route_construction.acceptance_criteria.pareto_dominance`

```{py:module} src.policies.route_construction.acceptance_criteria.pareto_dominance
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.pareto_dominance
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ParetoDominanceAcceptance <src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance
    :summary:
    ```
````

### API

`````{py:class} ParetoDominanceAcceptance
:canonical: src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.pareto_dominance.ParetoDominanceAcceptance.get_state

````

`````
