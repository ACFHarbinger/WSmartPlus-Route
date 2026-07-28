# {py:mod}`src.policies.mandatory_selection.base.selection_registry`

```{py:module} src.policies.mandatory_selection.base.selection_registry
```

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MandatorySelectionRegistry <src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry>`
  - ```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry
    :summary:
    ```
````

### API

`````{py:class} MandatorySelectionRegistry
:canonical: src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry._strategies
:type: typing.Dict[str, typing.Type[logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy]]
:value: >
   None

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry._strategies
```

````

````{py:method} register(name: str)
:canonical: src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry.register
```

````

````{py:method} get_strategy_class(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.mandatory_selection.IMandatorySelectionStrategy]]
:canonical: src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry.get_strategy_class
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry.get_strategy_class
```

````

````{py:method} list_strategies() -> list
:canonical: src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry.list_strategies
:classmethod:

```{autodoc2-docstring} src.policies.mandatory_selection.base.selection_registry.MandatorySelectionRegistry.list_strategies
```

````

`````
