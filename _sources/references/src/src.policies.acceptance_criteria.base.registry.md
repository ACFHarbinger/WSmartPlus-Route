# {py:mod}`src.policies.acceptance_criteria.base.registry`

```{py:module} src.policies.acceptance_criteria.base.registry
```

```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AcceptanceCriterionRegistry <src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry>`
  - ```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry
    :summary:
    ```
````

### API

`````{py:class} AcceptanceCriterionRegistry
:canonical: src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry

```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry
```

````{py:attribute} _registry
:canonical: src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry._registry
:type: typing.Dict[str, typing.Type[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]]
:value: >
   None

```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry._registry
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.acceptance_criterion.IAcceptanceCriterion]]
:canonical: src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry.get
```

````

````{py:method} list_criteria() -> typing.List[str]
:canonical: src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry.list_criteria
:classmethod:

```{autodoc2-docstring} src.policies.acceptance_criteria.base.registry.AcceptanceCriterionRegistry.list_criteria
```

````

`````
