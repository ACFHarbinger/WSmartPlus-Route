# {py:mod}`src.policies.selection_and_construction.base.registry`

```{py:module} src.policies.selection_and_construction.base.registry
```

```{autodoc2-docstring} src.policies.selection_and_construction.base.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`JointPolicyRegistry <src.policies.selection_and_construction.base.registry.JointPolicyRegistry>`
  - ```{autodoc2-docstring} src.policies.selection_and_construction.base.registry.JointPolicyRegistry
    :summary:
    ```
````

### API

`````{py:class} JointPolicyRegistry
:canonical: src.policies.selection_and_construction.base.registry.JointPolicyRegistry

```{autodoc2-docstring} src.policies.selection_and_construction.base.registry.JointPolicyRegistry
```

````{py:attribute} _registry
:canonical: src.policies.selection_and_construction.base.registry.JointPolicyRegistry._registry
:type: typing.Dict[str, typing.Type]
:value: >
   None

```{autodoc2-docstring} src.policies.selection_and_construction.base.registry.JointPolicyRegistry._registry
```

````

````{py:method} register(name: str)
:canonical: src.policies.selection_and_construction.base.registry.JointPolicyRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.base.registry.JointPolicyRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type]
:canonical: src.policies.selection_and_construction.base.registry.JointPolicyRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.base.registry.JointPolicyRegistry.get
```

````

````{py:method} list_policies() -> typing.List[str]
:canonical: src.policies.selection_and_construction.base.registry.JointPolicyRegistry.list_policies
:classmethod:

```{autodoc2-docstring} src.policies.selection_and_construction.base.registry.JointPolicyRegistry.list_policies
```

````

`````
