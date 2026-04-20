# {py:mod}`src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing`

```{py:module} src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`StepCountingHillClimbing <src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing
    :summary:
    ```
````

### API

`````{py:class} StepCountingHillClimbing(step_size: int)
:canonical: src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.step_counting_hill_climbing.StepCountingHillClimbing.get_state

````

`````
