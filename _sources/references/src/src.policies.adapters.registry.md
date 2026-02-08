# {py:mod}`src.policies.adapters.registry`

```{py:module} src.policies.adapters.registry
```

```{autodoc2-docstring} src.policies.adapters.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyRegistry <src.policies.adapters.registry.PolicyRegistry>`
  - ```{autodoc2-docstring} src.policies.adapters.registry.PolicyRegistry
    :summary:
    ```
````

### API

`````{py:class} PolicyRegistry
:canonical: src.policies.adapters.registry.PolicyRegistry

```{autodoc2-docstring} src.policies.adapters.registry.PolicyRegistry
```

````{py:attribute} _registry
:canonical: src.policies.adapters.registry.PolicyRegistry._registry
:type: typing.Dict[str, typing.Type[logic.src.interfaces.policy.IPolicy]]
:value: >
   None

```{autodoc2-docstring} src.policies.adapters.registry.PolicyRegistry._registry
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.adapters.registry.PolicyRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.adapters.registry.PolicyRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.policy.IPolicy]]
:canonical: src.policies.adapters.registry.PolicyRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.adapters.registry.PolicyRegistry.get
```

````

````{py:method} list_policies() -> typing.List[str]
:canonical: src.policies.adapters.registry.PolicyRegistry.list_policies
:classmethod:

```{autodoc2-docstring} src.policies.adapters.registry.PolicyRegistry.list_policies
```

````

`````
