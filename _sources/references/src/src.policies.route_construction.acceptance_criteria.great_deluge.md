# {py:mod}`src.policies.route_construction.acceptance_criteria.great_deluge`

```{py:module} src.policies.route_construction.acceptance_criteria.great_deluge
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.great_deluge
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GreatDelugeAcceptance <src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance
    :summary:
    ```
````

### API

`````{py:class} GreatDelugeAcceptance(target_fitness_multiplier: float, max_iterations: int)
:canonical: src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.great_deluge.GreatDelugeAcceptance.get_state

````

`````
