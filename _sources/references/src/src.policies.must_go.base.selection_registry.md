# {py:mod}`src.policies.must_go.base.selection_registry`

```{py:module} src.policies.must_go.base.selection_registry
```

```{autodoc2-docstring} src.policies.must_go.base.selection_registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`MustGoSelectionRegistry <src.policies.must_go.base.selection_registry.MustGoSelectionRegistry>`
  - ```{autodoc2-docstring} src.policies.must_go.base.selection_registry.MustGoSelectionRegistry
    :summary:
    ```
````

### API

`````{py:class} MustGoSelectionRegistry
:canonical: src.policies.must_go.base.selection_registry.MustGoSelectionRegistry

```{autodoc2-docstring} src.policies.must_go.base.selection_registry.MustGoSelectionRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.must_go.base.selection_registry.MustGoSelectionRegistry._strategies
:type: typing.Dict[str, typing.Type[logic.src.interfaces.must_go.MustGoSelectionStrategy]]
:value: >
   None

```{autodoc2-docstring} src.policies.must_go.base.selection_registry.MustGoSelectionRegistry._strategies
```

````

````{py:method} register(name: str)
:canonical: src.policies.must_go.base.selection_registry.MustGoSelectionRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.must_go.base.selection_registry.MustGoSelectionRegistry.register
```

````

````{py:method} get_strategy_class(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.must_go.MustGoSelectionStrategy]]
:canonical: src.policies.must_go.base.selection_registry.MustGoSelectionRegistry.get_strategy_class
:classmethod:

```{autodoc2-docstring} src.policies.must_go.base.selection_registry.MustGoSelectionRegistry.get_strategy_class
```

````

`````
