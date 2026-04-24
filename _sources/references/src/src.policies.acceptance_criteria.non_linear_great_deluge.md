# {py:mod}`src.policies.acceptance_criteria.non_linear_great_deluge`

```{py:module} src.policies.acceptance_criteria.non_linear_great_deluge
```

```{autodoc2-docstring} src.policies.acceptance_criteria.non_linear_great_deluge
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`NonLinearGreatDeluge <src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge
    :summary:
    ```
````

### API

`````{py:class} NonLinearGreatDeluge(t_max: int, initial_tolerance: float = 0.1, gap_epsilon: float = 0.01, beta: float = 5.0, maximization: bool = False)
:canonical: src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge.setup

````

````{py:method} _get_f_best(**kwargs: typing.Any) -> float
:canonical: src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge._get_f_best

```{autodoc2-docstring} src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge._get_f_best
```

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge.accept

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge.step

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.non_linear_great_deluge.NonLinearGreatDeluge.get_state

````

`````
