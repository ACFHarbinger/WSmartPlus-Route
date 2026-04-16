# {py:mod}`src.policies.route_construction.acceptance_criteria.fitness_proportional`

```{py:module} src.policies.route_construction.acceptance_criteria.fitness_proportional
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.fitness_proportional
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`FitnessProportionalAcceptance <src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance
    :summary:
    ```
````

### API

`````{py:class} FitnessProportionalAcceptance(seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.fitness_proportional.FitnessProportionalAcceptance.get_state

````

`````
