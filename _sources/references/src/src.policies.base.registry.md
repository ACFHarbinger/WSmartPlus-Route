# {py:mod}`src.policies.base.registry`

```{py:module} src.policies.base.registry
```

```{autodoc2-docstring} src.policies.base.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyRegistry <src.policies.base.registry.PolicyRegistry>`
  - ```{autodoc2-docstring} src.policies.base.registry.PolicyRegistry
    :summary:
    ```
````

### API

`````{py:class} PolicyRegistry
:canonical: src.policies.base.registry.PolicyRegistry

```{autodoc2-docstring} src.policies.base.registry.PolicyRegistry
```

````{py:attribute} _registry
:canonical: src.policies.base.registry.PolicyRegistry._registry
:type: typing.Dict[str, typing.Type[logic.src.interfaces.policy.IPolicy]]
:value: >
   None

```{autodoc2-docstring} src.policies.base.registry.PolicyRegistry._registry
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.base.registry.PolicyRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.base.registry.PolicyRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.policy.IPolicy]]
:canonical: src.policies.base.registry.PolicyRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.base.registry.PolicyRegistry.get
```

````

````{py:method} list_policies() -> typing.List[str]
:canonical: src.policies.base.registry.PolicyRegistry.list_policies
:classmethod:

```{autodoc2-docstring} src.policies.base.registry.PolicyRegistry.list_policies
```

````

`````
