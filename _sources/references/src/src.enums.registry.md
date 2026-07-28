# {py:mod}`src.enums.registry`

```{py:module} src.enums.registry
```

```{autodoc2-docstring} src.enums.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`GlobalRegistry <src.enums.registry.GlobalRegistry>`
  - ```{autodoc2-docstring} src.enums.registry.GlobalRegistry
    :summary:
    ```
````

### Data

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`AnyTag <src.enums.registry.AnyTag>`
  - ```{autodoc2-docstring} src.enums.registry.AnyTag
    :summary:
    ```
* - {py:obj}`T_Algorithm <src.enums.registry.T_Algorithm>`
  - ```{autodoc2-docstring} src.enums.registry.T_Algorithm
    :summary:
    ```
* - {py:obj}`T <src.enums.registry.T>`
  - ```{autodoc2-docstring} src.enums.registry.T
    :summary:
    ```
````

### API

````{py:data} AnyTag
:canonical: src.enums.registry.AnyTag
:value: >
   None

```{autodoc2-docstring} src.enums.registry.AnyTag
```

````

````{py:data} T_Algorithm
:canonical: src.enums.registry.T_Algorithm
:value: >
   None

```{autodoc2-docstring} src.enums.registry.T_Algorithm
```

````

````{py:data} T
:canonical: src.enums.registry.T
:value: >
   'TypeVar(...)'

```{autodoc2-docstring} src.enums.registry.T
```

````

`````{py:class} GlobalRegistry
:canonical: src.enums.registry.GlobalRegistry

```{autodoc2-docstring} src.enums.registry.GlobalRegistry
```

````{py:attribute} _registry
:canonical: src.enums.registry.GlobalRegistry._registry
:type: typing.Dict[src.enums.registry.T_Algorithm, typing.Set[src.enums.registry.AnyTag]]
:value: >
   None

```{autodoc2-docstring} src.enums.registry.GlobalRegistry._registry
```

````

````{py:method} register(*tags: src.enums.registry.AnyTag) -> typing.Callable[[src.enums.registry.T_Algorithm], src.enums.registry.T_Algorithm]
:canonical: src.enums.registry.GlobalRegistry.register
:classmethod:

```{autodoc2-docstring} src.enums.registry.GlobalRegistry.register
```

````

````{py:method} get_name(obj: src.enums.registry.T_Algorithm) -> str
:canonical: src.enums.registry.GlobalRegistry.get_name
:classmethod:

```{autodoc2-docstring} src.enums.registry.GlobalRegistry.get_name
```

````

````{py:method} query_intersection(*tags: src.enums.registry.AnyTag, expected_type: typing.Optional[typing.Type[src.enums.registry.T]] = None) -> typing.Union[typing.List[src.enums.registry.T_Algorithm], typing.List[typing.Type[src.enums.registry.T]]]
:canonical: src.enums.registry.GlobalRegistry.query_intersection
:classmethod:

```{autodoc2-docstring} src.enums.registry.GlobalRegistry.query_intersection
```

````

````{py:method} query_union(*tags: src.enums.registry.AnyTag, expected_type: typing.Optional[typing.Type[src.enums.registry.T]] = None) -> typing.Union[typing.List[src.enums.registry.T_Algorithm], typing.List[typing.Type[src.enums.registry.T]]]
:canonical: src.enums.registry.GlobalRegistry.query_union
:classmethod:

```{autodoc2-docstring} src.enums.registry.GlobalRegistry.query_union
```

````

````{py:method} query_difference(require: typing.List[src.enums.registry.AnyTag], exclude: typing.List[src.enums.registry.AnyTag], expected_type: typing.Optional[typing.Type[src.enums.registry.T]] = None) -> typing.Union[typing.List[src.enums.registry.T_Algorithm], typing.List[typing.Type[src.enums.registry.T]]]
:canonical: src.enums.registry.GlobalRegistry.query_difference
:classmethod:

```{autodoc2-docstring} src.enums.registry.GlobalRegistry.query_difference
```

````

````{py:method} get_all() -> typing.Dict[src.enums.registry.T_Algorithm, typing.Set[src.enums.registry.AnyTag]]
:canonical: src.enums.registry.GlobalRegistry.get_all
:classmethod:

```{autodoc2-docstring} src.enums.registry.GlobalRegistry.get_all
```

````

`````
