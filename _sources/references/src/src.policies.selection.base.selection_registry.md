# {py:mod}`src.policies.selection.base.selection_registry`

```{py:module} src.policies.selection.base.selection_registry
```

```{autodoc2-docstring} src.policies.selection.base.selection_registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MustGoSelectionRegistry <src.policies.selection.base.selection_registry.MustGoSelectionRegistry>`
  - ```{autodoc2-docstring} src.policies.selection.base.selection_registry.MustGoSelectionRegistry
    :summary:
    ```
````

### API

`````{py:class} MustGoSelectionRegistry
:canonical: src.policies.selection.base.selection_registry.MustGoSelectionRegistry

```{autodoc2-docstring} src.policies.selection.base.selection_registry.MustGoSelectionRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.selection.base.selection_registry.MustGoSelectionRegistry._strategies
:type: typing.Dict[str, typing.Type[src.policies.selection.base.selection_strategy.MustGoSelectionStrategy]]
:value: >
   None

```{autodoc2-docstring} src.policies.selection.base.selection_registry.MustGoSelectionRegistry._strategies
```

````

````{py:method} register(name: str)
:canonical: src.policies.selection.base.selection_registry.MustGoSelectionRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.selection.base.selection_registry.MustGoSelectionRegistry.register
```

````

````{py:method} get_strategy_class(name: str) -> typing.Optional[typing.Type[src.policies.selection.base.selection_strategy.MustGoSelectionStrategy]]
:canonical: src.policies.selection.base.selection_registry.MustGoSelectionRegistry.get_strategy_class
:classmethod:

```{autodoc2-docstring} src.policies.selection.base.selection_registry.MustGoSelectionRegistry.get_strategy_class
```

````

`````
