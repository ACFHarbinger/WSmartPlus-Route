# {py:mod}`src.policies.other.post_processing.registry`

```{py:module} src.policies.other.post_processing.registry
```

```{autodoc2-docstring} src.policies.other.post_processing.registry
:allowtitles:
```

## Module Contents

### Classes

````{list-table}
:class: autosummary longtable
:align: left

* - {py:obj}`PostProcessorRegistry <src.policies.other.post_processing.registry.PostProcessorRegistry>`
  - ```{autodoc2-docstring} src.policies.other.post_processing.registry.PostProcessorRegistry
    :summary:
    ```
````

### API

`````{py:class} PostProcessorRegistry
:canonical: src.policies.other.post_processing.registry.PostProcessorRegistry

```{autodoc2-docstring} src.policies.other.post_processing.registry.PostProcessorRegistry
```

````{py:attribute} _strategies
:canonical: src.policies.other.post_processing.registry.PostProcessorRegistry._strategies
:type: typing.Dict[str, typing.Type[logic.src.interfaces.post_processing.IPostProcessor]]
:value: >
   None

```{autodoc2-docstring} src.policies.other.post_processing.registry.PostProcessorRegistry._strategies
```

````

````{py:method} register(name: str) -> typing.Callable
:canonical: src.policies.other.post_processing.registry.PostProcessorRegistry.register
:classmethod:

```{autodoc2-docstring} src.policies.other.post_processing.registry.PostProcessorRegistry.register
```

````

````{py:method} get(name: str) -> typing.Optional[typing.Type[logic.src.interfaces.post_processing.IPostProcessor]]
:canonical: src.policies.other.post_processing.registry.PostProcessorRegistry.get
:classmethod:

```{autodoc2-docstring} src.policies.other.post_processing.registry.PostProcessorRegistry.get
```

````

`````
