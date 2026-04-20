# {py:mod}`src.policies.route_construction.acceptance_criteria.record_to_record_travel`

```{py:module} src.policies.route_construction.acceptance_criteria.record_to_record_travel
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.record_to_record_travel
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`RecordToRecordTravel <src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel
    :summary:
    ```
````

### API

`````{py:class} RecordToRecordTravel(tolerance: float)
:canonical: src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.record_to_record_travel.RecordToRecordTravel.get_state

````

`````
