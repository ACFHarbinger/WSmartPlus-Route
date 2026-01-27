# {py:mod}`src.policies.adapters`

```{py:module} src.policies.adapters
```

```{autodoc2-docstring} src.policies.adapters
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`IPolicy <src.policies.adapters.IPolicy>`
  - ```{autodoc2-docstring} src.policies.adapters.IPolicy
    :summary:
    ```
* - {py:obj}`PolicyRegistry <src.policies.adapters.PolicyRegistry>`
  - ```{autodoc2-docstring} src.policies.adapters.PolicyRegistry
    :summary:
    ```
* - {py:obj}`PolicyFactory <src.policies.adapters.PolicyFactory>`
  - ```{autodoc2-docstring} src.policies.adapters.PolicyFactory
    :summary:
    ```
````

### Functions

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`__getattr__ <src.policies.adapters.__getattr__>`
  - ```{autodoc2-docstring} src.policies.adapters.__getattr__
    :summary:
    ```
````

### API

`````{py:class} IPolicy
:canonical: src.policies.adapters.IPolicy

Bases: {py:obj}`abc.ABC`

```{autodoc2-docstring} src.policies.adapters.IPolicy
```

````{py:method} execute(**kwargs: typing.Any) -> typing.Tuple[typing.List[int], float, typing.Any]
:canonical: src.policies.adapters.IPolicy.execute
:abstractmethod:

```{autodoc2-docstring} src.policies.adapters.IPolicy.execute
```

````

`````

`````{py:class} PolicyRegistry
:canonical: src.policies.adapters.PolicyRegistry

```{autodoc2-docstring} src.policies.adapters.PolicyRegistry
```

````{py:attribute} _registry
:canonical: src.policies.adapters.PolicyRegistry._registry
:type: typing.Dict[str, typing.Type[src.policies.adapters.IPolicy]]
:value: >
   None

```{autodoc2-docstring} src.policies.adapters.PolicyRegistry._registry
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.adapters.PolicyRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.adapters.PolicyRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[src.policies.adapters.IPolicy]]
:canonical: src.policies.adapters.PolicyRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.adapters.PolicyRegistry.get
```

````

````{py:method} list_policies() -> typing.List[str]
:canonical: src.policies.adapters.PolicyRegistry.list_policies
:classmethod:

```{autodoc2-docstring} src.policies.adapters.PolicyRegistry.list_policies
```

````

`````

`````{py:class} PolicyFactory
:canonical: src.policies.adapters.PolicyFactory

```{autodoc2-docstring} src.policies.adapters.PolicyFactory
```

````{py:method} get_adapter(name: str, engine: typing.Optional[str] = None, threshold: typing.Optional[float] = None, **kwargs: typing.Any) -> src.policies.adapters.IPolicy
:canonical: src.policies.adapters.PolicyFactory.get_adapter
:staticmethod:

```{autodoc2-docstring} src.policies.adapters.PolicyFactory.get_adapter
```

````

`````

````{py:function} __getattr__(name: str) -> typing.Any
:canonical: src.policies.adapters.__getattr__

```{autodoc2-docstring} src.policies.adapters.__getattr__
```
````
