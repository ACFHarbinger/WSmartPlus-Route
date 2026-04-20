# {py:mod}`src.policies.route_construction.acceptance_criteria.monte_carlo`

```{py:module} src.policies.route_construction.acceptance_criteria.monte_carlo
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.monte_carlo
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MonteCarloAcceptance <src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance
    :summary:
    ```
````

### API

`````{py:class} MonteCarloAcceptance(p: float = 0.1, seed: typing.Optional[int] = None)
:canonical: src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.monte_carlo.MonteCarloAcceptance.get_state

````

`````
