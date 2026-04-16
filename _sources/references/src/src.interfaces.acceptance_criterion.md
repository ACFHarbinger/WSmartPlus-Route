# {py:mod}`src.interfaces.acceptance_criterion`

```{py:module} src.interfaces.acceptance_criterion
```

```{autodoc2-docstring} src.interfaces.acceptance_criterion
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IAcceptanceCriterion <src.interfaces.acceptance_criterion.IAcceptanceCriterion>`
  - ```{autodoc2-docstring} src.interfaces.acceptance_criterion.IAcceptanceCriterion
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`ObjectiveValue <src.interfaces.acceptance_criterion.ObjectiveValue>`
  - ```{autodoc2-docstring} src.interfaces.acceptance_criterion.ObjectiveValue
    :summary:
    ```
````

### API

````{py:data} ObjectiveValue
:canonical: src.interfaces.acceptance_criterion.ObjectiveValue
:value: >
   None

```{autodoc2-docstring} src.interfaces.acceptance_criterion.ObjectiveValue
```

````

`````{py:class} IAcceptanceCriterion
:canonical: src.interfaces.acceptance_criterion.IAcceptanceCriterion

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.interfaces.acceptance_criterion.IAcceptanceCriterion
```

````{py:method} setup(initial_objective: src.interfaces.acceptance_criterion.ObjectiveValue) -> None
:canonical: src.interfaces.acceptance_criterion.IAcceptanceCriterion.setup
:abstractmethod:

```{autodoc2-docstring} src.interfaces.acceptance_criterion.IAcceptanceCriterion.setup
```

````

````{py:method} accept(current_obj: src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: src.interfaces.acceptance_criterion.ObjectiveValue, **kwargs: typing.Any) -> typing.Tuple[bool, logic.src.policies.context.search_context.AcceptanceMetrics]
:canonical: src.interfaces.acceptance_criterion.IAcceptanceCriterion.accept
:abstractmethod:

```{autodoc2-docstring} src.interfaces.acceptance_criterion.IAcceptanceCriterion.accept
```

````

````{py:method} step(current_obj: src.interfaces.acceptance_criterion.ObjectiveValue, candidate_obj: src.interfaces.acceptance_criterion.ObjectiveValue, accepted: bool, **kwargs: typing.Any) -> None
:canonical: src.interfaces.acceptance_criterion.IAcceptanceCriterion.step
:abstractmethod:

```{autodoc2-docstring} src.interfaces.acceptance_criterion.IAcceptanceCriterion.step
```

````

````{py:method} get_state() -> typing.Dict[str, typing.Any]
:canonical: src.interfaces.acceptance_criterion.IAcceptanceCriterion.get_state

```{autodoc2-docstring} src.interfaces.acceptance_criterion.IAcceptanceCriterion.get_state
```

````

`````
