# {py:mod}`src.policies.acceptance_criteria.epsilon_dominance`

```{py:module} src.policies.acceptance_criteria.epsilon_dominance
```

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`EpsilonDominanceCriterion <src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion
    :summary:
    ```
````

### API

`````{py:class} EpsilonDominanceCriterion(epsilon: typing.Union[float, typing.Sequence[float]], maximization: bool = True)
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion

Bases: {py:obj}`logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion`

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion
```

```{rubric} Initialization
```

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.__init__
```

````{py:method} setup(initial_objective: logic.src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.setup

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.setup
```

````

````{py:method} _get_epsilon_box(objs: typing.Sequence[float]) -> typing.Tuple[int, ...]
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion._get_epsilon_box

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion._get_epsilon_box
```

````

````{py:method} _dominates(box_a: typing.Tuple[int, ...], box_b: typing.Tuple[int, ...]) -> bool
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion._dominates

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion._dominates
```

````

````{py:method} accept(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.interfaces.context.search_context.AcceptanceMetrics]
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.accept

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.accept
```

````

````{py:method} step(current_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: logic.src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.step

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.step
```

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.get_state

```{autodoc2-docstring} src.policies.acceptance_criteria.epsilon_dominance.EpsilonDominanceCriterion.get_state
```

````

`````
