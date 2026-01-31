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

* - {py:obj}`IPolicy <src.policies.adapters.factory.IPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.factory.IPolicy
    :summary:
    ```
* - {py:obj}`PolicyRegistry <src.policies.adapters.factory.PolicyRegistry>`
  - ```{autodoc2-docstring} src.policies.adapters.factory.PolicyRegistry
    :summary:
    ```
* - {py:obj}`PolicyFactory <src.policies.adapters.factory.PolicyFactory>`
  - ```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__getattr__ <src.policies.adapters.factory.__getattr__>`
  - ```{autodoc2-docstring} src.policies.adapters.factory.__getattr__
    :summary:
    ```
````

### API

`````{py:class} IPolicy
:canonical: src.policies.adapters.factory.IPolicy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.adapters.factory.IPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.adapters.factory.IPolicy.execute
:abstractmethod:

```{autodoc2-docstring} src.policies.adapters.factory.IPolicy.execute
```

````

`````

`````{py:class} PolicyRegistry
:canonical: src.policies.adapters.factory.PolicyRegistry

```{autodoc2-docstring} src.policies.adapters.factory.PolicyRegistry
```

````{py:attribute} _registry
:canonical: src.policies.adapters.factory.PolicyRegistry._registry
:type: typing.Dict[str, typing.Type[src.policies.adapters.factory.IPolicy]]
:value: >
   None

```{autodoc2-docstring} src.policies.adapters.factory.PolicyRegistry._registry
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.adapters.factory.PolicyRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.adapters.factory.PolicyRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[src.policies.adapters.factory.IPolicy]]
:canonical: src.policies.adapters.factory.PolicyRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.adapters.factory.PolicyRegistry.get
```

````

````{py:method} list_policies() -> typing.List[str]
:canonical: src.policies.adapters.factory.PolicyRegistry.list_policies
:classmethod:

```{autodoc2-docstring} src.policies.adapters.factory.PolicyRegistry.list_policies
```

````

`````

`````{py:class} PolicyFactory
:canonical: src.policies.adapters.factory.PolicyFactory

```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory
```

````{py:method} get_adapter(name: str, engine: typing.Optional[str] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> src.policies.adapters.factory.IPolicy
:canonical: src.policies.adapters.factory.PolicyFactory.get_adapter
:staticmethod:

```{autodoc2-docstring} src.policies.adapters.factory.PolicyFactory.get_adapter
```

````

`````

````{py:function} __getattr__(name: str) -> typing.Any
:canonical: src.policies.adapters.factory.__getattr__

```{autodoc2-docstring} src.policies.adapters.factory.__getattr__
```
````
