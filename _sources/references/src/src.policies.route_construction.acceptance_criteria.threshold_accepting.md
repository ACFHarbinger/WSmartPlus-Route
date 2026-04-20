# {py:mod}`src.policies.route_construction.acceptance_criteria.threshold_accepting`

```{py:module} src.policies.route_construction.acceptance_criteria.threshold_accepting
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.threshold_accepting
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ThresholdAccepting <src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting
    :summary:
    ```
````

### API

`````{py:class} ThresholdAccepting(initial_threshold: float, max_iterations: int)
:canonical: src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.threshold_accepting.ThresholdAccepting.get_state

````

`````
