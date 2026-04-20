# {py:mod}`src.policies.route_construction.acceptance_criteria.demon_algorithm`

```{py:module} src.policies.route_construction.acceptance_criteria.demon_algorithm
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.demon_algorithm
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`DemonAlgorithm <src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm>`
  - ```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm
    :summary:
    ```
````

### API

`````{py:class} DemonAlgorithm(warm_up_steps: int = 5, maximization: bool = False)
:canonical: src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm.setup

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.route_construction.acceptance_criteria.demon_algorithm.DemonAlgorithm.get_state

````

`````
