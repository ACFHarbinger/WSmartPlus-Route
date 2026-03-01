# {py:mod}`src.policies.adapters.factory`

```{py:module} src.policies.adapters.factory
```

```{autodoc2-docstring} src.policies.adapters.factory
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PolicyFactory <src.policies.adapters.factory.PolicyFactory>`
  - ```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPolicy <src.policies.adapters.factory.IPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.factory.IPolicy
    :summary:
    ```
````

### API

````{py:data} IPolicy
:canonical: src.policies.adapters.factory.IPolicy
:value: >
   None

```{autodoc2-docstring} src.policies.adapters.factory.IPolicy
```

````

`````{py:class} PolicyFactory
:canonical: src.policies.adapters.factory.PolicyFactory

```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory
```

````{py:attribute} _registered
:canonical: src.policies.adapters.factory.PolicyFactory._registered
:value: >
   False

```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory._registered
```

````

````{py:method} ensure_registered() -> None
:canonical: src.policies.adapters.factory.PolicyFactory.ensure_registered
:classmethod:

```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory.ensure_registered
```

````

````{py:method} get_adapter(name: str, config: typing.Optional[dict] = None, engine: typing.Optional[str] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> src.policies.adapters.factory.IPolicy
:canonical: src.policies.adapters.factory.PolicyFactory.get_adapter
:staticmethod:

```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory.get_adapter
```

````

`````
