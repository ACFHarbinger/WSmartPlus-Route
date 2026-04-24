# {py:mod}`src.policies.acceptance_criteria.skewed_variable_neighborhood_search`

```{py:module} src.policies.acceptance_criteria.skewed_variable_neighborhood_search
```

```{autodoc2-docstring} src.policies.acceptance_criteria.skewed_variable_neighborhood_search
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`SkewedVNSAcceptance <src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance
    :summary:
    ```
````

### API

`````{py:class} SkewedVNSAcceptance(alpha: float, metric: logic.src.interfaces.distance_metric.IDistanceMetric, maximization: bool = True)
:canonical: src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.skewed_variable_neighborhood_search.SkewedVNSAcceptance.get_state

````

`````
