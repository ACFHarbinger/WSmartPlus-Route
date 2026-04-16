# {py:mod}`src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion`

```{py:module} src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`BoltzmannAcceptance <src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance
    :summary:
    ```
````

### API

`````{py:class} BoltzmannAcceptance(initial_temp: float, alpha: float, seed: typing.Optional[int] = 42)
:canonical: src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.boltzmann_metropolis_criterion.BoltzmannAcceptance.get_state

````

`````
