# {py:mod}`src.policies.base.factory`

```{py:module} src.policies.base.factory
```

```{autodoc2-docstring} src.policies.base.factory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyFactory <src.policies.base.factory.PolicyFactory>`
  - ```{autodoc2-docstring} src.policies.base.factory.PolicyFactory
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPolicy <src.policies.base.factory.IPolicy>`
  - ```{autodoc2-docstring} src.policies.base.factory.IPolicy
    :summary:
    ```
````

### API

````{py:data} IPolicy
:canonical: src.policies.base.factory.IPolicy
:value: >
   None

```{autodoc2-docstring} src.policies.base.factory.IPolicy
```

````

`````{py:class} PolicyFactory
:canonical: src.policies.base.factory.PolicyFactory

```{autodoc2-docstring} src.policies.base.factory.PolicyFactory
```

````{py:attribute} _registered
:canonical: src.policies.base.factory.PolicyFactory._registered
:value: >
   False

```{autodoc2-docstring} src.policies.base.factory.PolicyFactory._registered
```

````

````{py:method} ensure_registered() -> None
:canonical: src.policies.base.factory.PolicyFactory.ensure_registered
:classmethod:

```{autodoc2-docstring} src.policies.base.factory.PolicyFactory.ensure_registered
```

````

````{py:method} get_adapter(name: str, config: typing.Optional[dict] = None, engine: typing.Optional[str] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> src.policies.base.factory.IPolicy
:canonical: src.policies.base.factory.PolicyFactory.get_adapter
:staticmethod:

```{autodoc2-docstring} src.policies.base.factory.PolicyFactory.get_adapter
```

````

`````
