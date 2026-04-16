# {py:mod}`src.policies.route_construction.acceptance_criteria.probabilistic_transition`

```{py:module} src.policies.route_construction.acceptance_criteria.probabilistic_transition
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.probabilistic_transition
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ProbabilisticTransitionAcceptance <src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance
    :summary:
    ```
````

### API

`````{py:class} ProbabilisticTransitionAcceptance(alpha: float = 1.0, seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.probabilistic_transition.ProbabilisticTransitionAcceptance.get_state

````

`````
